-- Flux Memory v0.5 — SQLite schema (Section 6.4 of the spec).
-- Executed by FluxStore at startup. Idempotent via IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS grains (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    provenance      TEXT NOT NULL,              -- user_stated | ai_stated | ai_inferred | external_source
    confidence      REAL NOT NULL DEFAULT 1.0,  -- source-weighted confidence, 0.0-1.0
    decay_class     TEXT NOT NULL DEFAULT 'working',  -- working | core
    status          TEXT NOT NULL DEFAULT 'active',   -- active | dormant | archived | quarantined | pending_deletion
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    dormant_since   TEXT,
    context_spread  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conduits (
    id          TEXT PRIMARY KEY,
    from_id     TEXT NOT NULL,
    to_id       TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.25,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_used   TEXT NOT NULL DEFAULT (datetime('now')),
    use_count   INTEGER NOT NULL DEFAULT 0,
    direction   TEXT NOT NULL DEFAULT 'forward',  -- forward | bidirectional
    decay_class TEXT NOT NULL DEFAULT 'working',  -- core | working | ephemeral
    UNIQUE(from_id, to_id)
);

CREATE TABLE IF NOT EXISTS entries (
    id          TEXT PRIMARY KEY,
    feature     TEXT NOT NULL UNIQUE,
    affinities  TEXT NOT NULL DEFAULT '{}'  -- JSON map of conduit_id -> affinity_float
);

CREATE TABLE IF NOT EXISTS entry_cluster_membership (
    entry_id     TEXT NOT NULL,
    cluster_id   TEXT NOT NULL,
    weight       REAL NOT NULL,              -- soft membership, per-entry weights sum to 1.0
    PRIMARY KEY (entry_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS entry_cooccurrence (
    entry_a      TEXT NOT NULL,
    entry_b      TEXT NOT NULL,
    count        INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (entry_a, entry_b)
);

CREATE TABLE IF NOT EXISTS clusters (
    id           TEXT PRIMARY KEY,
    size         INTEGER NOT NULL DEFAULT 0,  -- entry points with membership > 0.1
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_updated TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS grain_cluster_touch (
    grain_id     TEXT NOT NULL,
    cluster_id   TEXT NOT NULL,
    touch_weight REAL NOT NULL DEFAULT 0.0,
    last_touched TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (grain_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS traces (
    id                    TEXT PRIMARY KEY,
    query_text            TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    feedback_at           TEXT,
    hop_count             INTEGER,
    activated_grain_count INTEGER,
    trace_data            TEXT  -- JSON: full conduit path + signal values
);

CREATE TABLE IF NOT EXISTS co_retrieval_counts (
    grain_a     TEXT NOT NULL,              -- canonicalized: grain_a < grain_b
    grain_b     TEXT NOT NULL,
    count       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (grain_a, grain_b)
);

-- Structured event log (Section 11.5).
-- All operations emit events here. Also written to a rotating JSON log file.
CREATE TABLE IF NOT EXISTS events (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    category    TEXT NOT NULL,  -- retrieval|feedback|write|decay|cluster|health|system|admin
    event       TEXT NOT NULL,
    trace_id    TEXT,           -- optional link to traces.id
    data        TEXT NOT NULL DEFAULT '{}'  -- JSON blob per §11.5 event structure
);

-- Health signal snapshots (Section 12). One row per signal per computation run.
CREATE TABLE IF NOT EXISTS health_log (
    id          TEXT PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    signal      TEXT NOT NULL,
    value       REAL NOT NULL,
    healthy     INTEGER NOT NULL DEFAULT 1,  -- 1=healthy, 0=unhealthy
    window      TEXT NOT NULL DEFAULT 'short'  -- short|medium|long
);

-- Active warnings table (Section 12.2). Cleared when signal returns to healthy range.
CREATE TABLE IF NOT EXISTS warnings (
    id              TEXT PRIMARY KEY,
    signal          TEXT NOT NULL UNIQUE,
    severity        TEXT NOT NULL,  -- INFO|WARNING|CRITICAL
    current_value   REAL NOT NULL,
    healthy_range   TEXT NOT NULL,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    cleared_at      TEXT,
    suggestion      TEXT NOT NULL DEFAULT ''
);

-- Grain embeddings for bootstrap and vector fallback (Section 4.7, 4.8).
-- Stored as JSON-encoded float array. One row per active grain.
CREATE TABLE IF NOT EXISTS grain_embeddings (
    grain_id    TEXT PRIMARY KEY,
    embedding   TEXT NOT NULL,  -- JSON float array
    model_name  TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Propagation performance indexes (Section 6.4).
CREATE INDEX IF NOT EXISTS idx_conduits_from ON conduits(from_id) WHERE weight >= 0.05;
CREATE INDEX IF NOT EXISTS idx_conduits_to   ON conduits(to_id);
CREATE INDEX IF NOT EXISTS idx_grains_status ON grains(status);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);

-- Additional indexes not named in §6.4 but implied by access patterns.
CREATE INDEX IF NOT EXISTS idx_conduits_last_used ON conduits(last_used);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category, timestamp);
CREATE INDEX IF NOT EXISTS idx_health_log_timestamp ON health_log(timestamp, signal);
