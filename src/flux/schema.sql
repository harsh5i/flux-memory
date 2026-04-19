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

-- Propagation performance indexes (Section 6.4).
CREATE INDEX IF NOT EXISTS idx_conduits_from ON conduits(from_id) WHERE weight >= 0.05;
CREATE INDEX IF NOT EXISTS idx_conduits_to   ON conduits(to_id);
CREATE INDEX IF NOT EXISTS idx_grains_status ON grains(status);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);

-- Additional indexes not named in §6.4 but implied by access patterns.
-- last_used drives the cleanup pass candidate query; without this it's a full scan.
CREATE INDEX IF NOT EXISTS idx_conduits_last_used ON conduits(last_used);
