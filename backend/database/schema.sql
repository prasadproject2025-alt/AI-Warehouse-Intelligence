CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    duration_sec REAL,
    fps REAL,
    frame_count INTEGER,
    width INTEGER,
    height INTEGER,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending',
    annotated_filepath TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    timestamp_sec REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    behaviour_type TEXT NOT NULL,
    object_track_id INTEGER,
    operator_track_id INTEGER,
    confidence REAL,
    risk_level TEXT NOT NULL,
    risk_score REAL NOT NULL,
    evidence_description TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    bounding_box TEXT NOT NULL, -- JSON array [x1, y1, x2, y2]
    evidence_image_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS frame_trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    frame_idx INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    track_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    box_json TEXT NOT NULL,
    vx REAL,
    vy REAL,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE INDEX IF NOT EXISTS idx_incidents_video ON incidents(video_id);
CREATE INDEX IF NOT EXISTS idx_incidents_risk ON incidents(risk_level);
CREATE INDEX IF NOT EXISTS idx_trajectories_video_frame ON frame_trajectories(video_id, frame_idx);
