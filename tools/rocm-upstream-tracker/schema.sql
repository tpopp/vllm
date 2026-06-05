CREATE TABLE IF NOT EXISTS sync_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    upstream_sha_before TEXT,
    upstream_sha_after TEXT,
    commits_count INTEGER DEFAULT 0,
    fork_push_success INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    message TEXT
);

CREATE TABLE IF NOT EXISTS upstream_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sync_run_id INTEGER,
    commit_sha TEXT NOT NULL UNIQUE,
    commit_date TEXT,
    author TEXT,
    commit_subject TEXT,
    pr_number INTEGER,
    pr_url TEXT,
    pr_title TEXT,
    summary TEXT,
    is_breaking_api INTEGER DEFAULT 0,
    categories TEXT,
    rocm_relevance TEXT,
    changed_files TEXT,
    heuristic_score REAL,
    action_hint TEXT,
    model_used TEXT,
    analysis_backend TEXT,
    analysis_cached_at TEXT,
    FOREIGN KEY (sync_run_id) REFERENCES sync_runs(id)
);

CREATE TABLE IF NOT EXISTS model_impacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    change_id INTEGER NOT NULL,
    architecture TEXT NOT NULL,
    module_name TEXT,
    impact_level TEXT NOT NULL,
    rationale TEXT,
    FOREIGN KEY (change_id) REFERENCES upstream_changes(id)
);

CREATE INDEX IF NOT EXISTS idx_model_arch ON model_impacts(architecture);
CREATE INDEX IF NOT EXISTS idx_change_sha ON upstream_changes(commit_sha);
CREATE INDEX IF NOT EXISTS idx_breaking ON upstream_changes(is_breaking_api);

CREATE VIEW IF NOT EXISTS v_breaking_by_model AS
SELECT c.commit_sha, c.commit_date, c.pr_url, c.summary, c.is_breaking_api,
       m.architecture, m.impact_level, m.rationale
FROM upstream_changes c
JOIN model_impacts m ON m.change_id = c.id
WHERE c.is_breaking_api = 1;

CREATE VIEW IF NOT EXISTS v_perf_opportunities AS
SELECT c.commit_sha, c.commit_date, c.pr_url, c.summary, c.categories,
       m.architecture, m.impact_level
FROM upstream_changes c
LEFT JOIN model_impacts m ON m.change_id = c.id
WHERE c.categories LIKE '%perf_immediate%'
   OR c.categories LIKE '%perf_with_work%';

CREATE VIEW IF NOT EXISTS v_nvidia_to_port AS
SELECT c.commit_sha, c.commit_date, c.pr_url, c.summary, c.action_hint,
       m.architecture, m.impact_level
FROM upstream_changes c
LEFT JOIN model_impacts m ON m.change_id = c.id
WHERE c.categories LIKE '%nvidia_replicate%';
