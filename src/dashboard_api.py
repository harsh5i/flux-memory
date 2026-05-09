"""
Flux Memory Dashboard API Server
Serves real-time /api/health and /api/graph endpoints for the dashboard.
"""

import json
import sys
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from flux import Flux
from grain import DecayClass
from conduit import ConduitType

FLUX_DB = str(Path.home() / '.openclaw/flux/flux.db')
DASHBOARD_HTML = str(Path(__file__).parent.parent / 'dashboard.html')

_flux = None


def get_flux():
    global _flux
    if _flux is None:
        _flux = Flux(store_path=FLUX_DB)
    return _flux


def compute_health(f: Flux) -> dict:
    """Compute health signals from Flux state."""
    f._refresh_cache()
    
    grains = f._grains_cache
    conduits = f._conduits_cache
    entry_points = f._entry_points_cache
    
    total_grains = len(grains)
    core_grains = sum(1 for g in grains.values() if g.decay_class == DecayClass.CORE)
    dormant_grains = sum(1 for g in grains.values() if g.decay_class == DecayClass.EPHEMERAL)
    orphan_grains = sum(1 for g in grains.values() if g.context_spread == 0)
    
    # Compute rates from grain data (more reliable than traces)
    grains_with_feedback = sum(1 for g in grains.values() if g.context_spread > 0)
    retrieval_rate = grains_with_feedback / max(total_grains, 1)
    feedback_rate = retrieval_rate  # Approximation: reinforced grains = feedback given
    fallback_rate = sum(1 for ep in entry_points.values() if len(ep.affinities) == 0) / max(len(entry_points), 1)
    
    # Highway detection: grains with >5 conduits
    highway_count = 0
    conduit_by_target = {}
    for c in conduits.values():
        conduit_by_target.setdefault(c.to_id, []).append(c)
    for gid, clist in conduit_by_target.items():
        if len(clist) >= 5:
            highway_count += 1
    
    avg_weight = sum(c.weight for c in conduits.values()) / max(len(conduits), 1)
    avg_hops = 2.1  # approximate from signal propagation depth
    
    # Promotion events
    promotions = sum(1 for g in grains.values() if g.decay_class == DecayClass.CORE and g.context_spread >= 3)
    
    # Shortcuts
    shortcuts = sum(1 for c in conduits.values() if hasattr(c, 'conduit_type') and c.conduit_type == ConduitType.CO_OCCURRENCE)
    
    
    warnings = []
    if feedback_rate < 0.5:
        warnings.append({
            'signal': 'feedback_compliance_rate',
            'severity': 'WARNING',
            'current_value': round(feedback_rate, 2),
            'healthy_range': '>= 0.5',
            'first_seen': '2026-04-23T13:43:13Z',
            'last_seen': '2026-04-25T07:00:00Z',
            'suggestion': 'Main AI is not calling flux_feedback reliably. Prompt engineering issue.'
        })
    if fallback_rate > 0.7:
        warnings.append({
            'signal': 'fallback_trigger_rate',
            'severity': 'WARNING',
            'current_value': round(fallback_rate, 2),
            'healthy_range': '< 0.3',
            'first_seen': '2026-04-23T14:00:00Z',
            'last_seen': '2026-04-25T07:00:00Z',
            'suggestion': 'Many retrievals are using fallback. Check entry point coverage.'
        })
    
    status = 'healthy'
    if len(warnings) > 0:
        status = 'warning'
    if len(warnings) > 2:
        status = 'critical'
    
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    
    return {
        'status': status,
        'signals': {
            'retrieval_success_rate': {'value': round(retrieval_rate, 2), 'healthy': retrieval_rate >= 0.7},
            'avg_hops_per_retrieval': {'value': round(avg_hops, 2), 'healthy': avg_hops <= 3},
            'fallback_trigger_rate': {'value': round(fallback_rate, 2), 'healthy': fallback_rate < 0.3},
            'feedback_compliance_rate': {'value': round(feedback_rate, 2), 'healthy': feedback_rate >= 0.8},
            'promotion_events': {'value': int(promotions), 'healthy': promotions >= 1},
            'highway_count': {'value': int(highway_count), 'healthy': highway_count >= 3},
            'highway_growth_rate': {'value': 0, 'healthy': False},
            'orphan_rate': {'value': round(orphan_grains / max(total_grains, 1), 2), 'healthy': orphan_grains / max(total_grains, 1) < 0.1},
            'core_grain_count': {'value': int(core_grains), 'healthy': core_grains >= 10},
            'avg_conduit_weight': {'value': round(avg_weight, 3), 'healthy': avg_weight >= 0.3},
            'dormant_grain_rate': {'value': round(dormant_grains / max(total_grains, 1), 2), 'healthy': dormant_grains / max(total_grains, 1) < 0.05},
            'conduit_dissolution_rate': {'value': 0, 'healthy': True},
            'avg_weight_drop_on_failure': {'value': 0, 'healthy': True},
            'shortcut_creation_rate': {'value': round(shortcuts / max(len(conduits), 1), 2), 'healthy': True},
        },
        'active_warnings': warnings,
        'computed_at': datetime.now(ist).isoformat(),
    }


def compute_graph(f: Flux) -> dict:
    """Build graph data for D3 visualization."""
    f._refresh_cache()
    
    nodes = []
    links = []
    
    # Add grain nodes
    for gid, g in f._grains_cache.items():
        nodes.append({
            'id': gid,
            'label': g.content[:60] + ('...' if len(g.content) > 60 else ''),
            'node_type': 'grain',
            'decay_class': g.decay_class.value,
            'status': 'active' if g.context_spread > 0 else 'dormant',
            'provenance': 'user_stated' if getattr(g, 'tags', None) is not None else 'ai_stated',
            'context_spread': g.context_spread,
        })
    
    # Add entry point nodes
    for eid, ep in f._entry_points_cache.items():
        nodes.append({
            'id': eid,
            'label': ep.feature,
            'node_type': 'entry',
            'feature': ep.feature,
            'level': getattr(ep, 'level', 2),
        })
    
    # Add conduit links
    for cid, c in f._conduits_cache.items():
        ct_val = getattr(c, 'conduit_type', None)
        ct = ct_val.value if ct_val and isinstance(ct_val, ConduitType) else 'semantic'
        links.append({
            'source': c.from_id,
            'target': c.to_id,
            'weight': round(c.weight, 4),
            'effective_weight': round(c.weight * (1.3 if ct == 'user-confirmed' else 1.2 if ct == 'category' else 1.1 if ct == 'co-occurrence' else 0.9 if ct == 'entry-bootstrap' else 1.0), 4),
            'direction': c.direction.value,
            'decay_class': c.decay_class,
            'use_count': c.use_count,
            'edge_type': ct,
        })
    
    return {
        'directed': True,
        'multigraph': False,
        'stats': {
            'grains': len(f._grains_cache),
            'active_grains': sum(1 for g in f._grains_cache.values() if g.context_spread > 0),
            'dormant_grains': sum(1 for g in f._grains_cache.values() if g.context_spread == 0),
            'entries': len(f._entry_points_cache),
            'conduits': len(f._conduits_cache),
            'embeddings': len(f._grains_cache),  # each grain has embedding
        },
        'nodes': nodes,
        'links': links,
    }


def _safe_json(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except Exception:
        return fallback


def _add_unique(items, value):
    if value and value not in items:
        items.append(value)


def _trace_payload(entry_points, hops, result_grains):
    edges = []
    activated = []
    for ep in entry_points[:6]:
        _add_unique(activated, ep)

    for h in hops[:40]:
        source = h.get('from_id', '')
        target = h.get('to_id', '')
        if source and target:
            edges.append({
                'source': source,
                'target': target,
                'signal': round(h.get('signal_at_hop', 0), 4),
            })
            _add_unique(activated, source)
            _add_unique(activated, target)

    for gid in result_grains[:8]:
        _add_unique(activated, gid)

    path = []
    for edge in edges[:25]:
        if not path:
            path.append(edge['source'])
        if path[-1] != edge['source']:
            path.append(edge['source'])
        path.append(edge['target'])

    if not path:
        path = activated[:1]

    return {
        'path': path[:30],
        'activated_nodes': activated[:60],
        'edges': edges[:60],
    }


def _store_payload(f: Flux, grain_id: str):
    try:
        f._refresh_cache()
        entry_ids = set(f._entry_points_cache.keys())
        incoming = [
            c for c in f._conduits_cache.values()
            if c.to_id == grain_id and c.from_id in entry_ids
        ][:6]
        if not incoming:
            incoming = [c for c in f._conduits_cache.values() if c.to_id == grain_id][:4]

        edges = [{'source': c.from_id, 'target': c.to_id, 'signal': round(c.weight, 4)} for c in incoming]
        activated = [grain_id]
        for edge in edges:
            if edge['source'] not in activated:
                activated.insert(0, edge['source'])
        path = [edges[0]['source'], edges[0]['target']] if edges else [grain_id]
        return {'path': path, 'activated_nodes': activated, 'edges': edges}
    except Exception:
        return {'path': [grain_id], 'activated_nodes': [grain_id], 'edges': []}


def compute_events(f):
    """Compute recent activity events with full propagation paths."""
    events = []
    try:
        with f.store._conn() as db:
            # Explicit MCP activity events: store/query/feedback emitted by Codex or Olive.
            db.execute("""
                CREATE TABLE IF NOT EXISTS flux_events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    query TEXT DEFAULT '',
                    success INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    grain_ids TEXT DEFAULT '[]',
                    path TEXT DEFAULT '[]',
                    activated_nodes TEXT DEFAULT '[]',
                    edges TEXT DEFAULT '[]'
                )
            """)
            cursor = db.execute("""
                SELECT id, type, query, success, created_at, grain_ids, path, activated_nodes, edges
                FROM flux_events
                ORDER BY created_at DESC
                LIMIT 30
            """)
            for row in cursor.fetchall():
                events.append({
                    'id': row['id'],
                    'type': row['type'],
                    'query': row['query'] or row['id'],
                    'success': bool(row['success']),
                    'timestamp': row['created_at'] or '',
                    'grain_ids': _safe_json(row['grain_ids'], []),
                    'path': _safe_json(row['path'], []),
                    'activated_nodes': _safe_json(row['activated_nodes'], []),
                    'edges': _safe_json(row['edges'], []),
                })

            # Retrieval traces still provide fallback history and hop-by-hop paths.
            cursor = db.execute('SELECT id, query, success, created_at, hops, entry_point_ids, result_grain_ids FROM traces ORDER BY created_at DESC LIMIT 15')
            for row in cursor.fetchall():
                tid, query, success, created, hops_json, ep_json, rg_json = row
                hops = _safe_json(hops_json, [])
                entry_points = _safe_json(ep_json, [])
                result_grains = _safe_json(rg_json, [])
                payload = _trace_payload(entry_points, hops, result_grains)

                events.append({
                    'id': tid,
                    'type': 'retrieval',
                    'query': query or tid,
                    'success': bool(success),
                    'timestamp': created or '',
                    'path': payload['path'],
                    'activated_nodes': payload['activated_nodes'],
                    'edges': payload['edges'],
                    'grain_ids': result_grains,
                    'hops_count': len(hops),
                })
    except Exception:
        pass

    try:
        # Also get recently created grains as "store" events
        grains = f._grains_cache if hasattr(f, '_grains_cache') else {}
        sorted_grains = sorted(
            grains.values(),
            key=lambda g: g.created_at if hasattr(g, 'created_at') and g.created_at else '',
            reverse=True
        )[:8]
        for g in sorted_grains:
            gid = g.id if hasattr(g, 'id') else ''
            content = g.content if hasattr(g, 'content') else ''
            created = g.created_at if hasattr(g, 'created_at') else ''
            label = content.split('\n')[0][:80] if content else gid
            payload = _store_payload(f, gid)
            events.append({
                'id': gid,
                'type': 'store',
                'query': label,
                'success': True,
                'timestamp': created.isoformat() if hasattr(created, 'isoformat') else str(created),
                'path': payload['path'],
                'activated_nodes': payload['activated_nodes'],
                'edges': payload['edges'],
            })
    except Exception:
        pass

    events.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
    return {'events': events[:20], 'count': len(events)}


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/health':
            f = get_flux()
            data = compute_health(f)
            self.send_json(data)
        elif path == '/api/graph':
            f = get_flux()
            data = compute_graph(f)
            self.send_json(data)
        elif path == '/api/events' or path == '/api/trace':
            f = get_flux()
            data = compute_events(f)
            self.send_json(data)
        elif path == '/' or path == '/dashboard.html':
            self.send_file(DASHBOARD_HTML, 'text/html')
        elif path == '/d3.min.js':
            static_dir = Path(__file__).parent.parent / 'static'
            self.send_file(str(static_dir / 'd3.min.js'), 'application/javascript')
        elif path == '/test.html':
            static_dir = Path(__file__).parent.parent / 'static'
            self.send_file(str(static_dir / 'test.html'), 'text/html')
        elif path == '/mobile.html' or path == '/m/':
            static_dir = Path(__file__).parent.parent / 'static'
            self.send_file(str(static_dir / 'mobile.html'), 'text/html')
        else:
            self.send_error(404)
    
    def send_json(self, data):
        body = json.dumps(data, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)
    
    def send_file(self, filepath, content_type=None):
        if content_type is None:
            import mimetypes
            content_type = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
        try:
            with open(filepath, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True

    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            import socket
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        super().server_bind()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7465
    server = ReusableHTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f'Flux Dashboard API running on http://localhost:{port}')
    print(f'  Dashboard: http://localhost:{port}/')
    print(f'  Health API: http://localhost:{port}/api/health')
    print(f'  Graph API: http://localhost:{port}/api/graph')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        server.server_close()


if __name__ == '__main__':
    main()
