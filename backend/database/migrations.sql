-- Additive migrations applied on every start. Each statement is executed
-- independently and "duplicate column" errors are ignored, so this file is
-- safe to run against both fresh and existing databases.

ALTER TABLE videos ADD COLUMN camera_id TEXT DEFAULT 'CAM-01';
ALTER TABLE videos ADD COLUMN bay TEXT DEFAULT 'Unassigned Bay';
ALTER TABLE videos ADD COLUMN shift TEXT DEFAULT 'Unassigned Shift';
ALTER TABLE videos ADD COLUMN recorded_at TIMESTAMP;
ALTER TABLE videos ADD COLUMN error_message TEXT;
ALTER TABLE videos ADD COLUMN processing_seconds REAL;
ALTER TABLE videos ADD COLUMN detector_backend TEXT;
ALTER TABLE videos ADD COLUMN frames_analysed INTEGER;
ALTER TABLE videos ADD COLUMN scene_flags TEXT;      -- JSON: declared scene context

ALTER TABLE incidents ADD COLUMN camera_id TEXT;
ALTER TABLE incidents ADD COLUMN bay TEXT;
ALTER TABLE incidents ADD COLUMN shift TEXT;
ALTER TABLE incidents ADD COLUMN risk_factors TEXT;   -- JSON: transparent score breakdown
ALTER TABLE incidents ADD COLUMN evidence_stages TEXT; -- JSON: temporal state sequence
ALTER TABLE incidents ADD COLUMN evidence_clip_path TEXT;
ALTER TABLE incidents ADD COLUMN evidence_tier TEXT DEFAULT 'OBSERVED_BEHAVIOUR';
ALTER TABLE incidents ADD COLUMN review_status TEXT DEFAULT 'PENDING_REVIEW';
ALTER TABLE incidents ADD COLUMN reviewer_note TEXT;
ALTER TABLE incidents ADD COLUMN reviewed_at TIMESTAMP;
ALTER TABLE incidents ADD COLUMN duration_sec REAL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_incidents_behaviour ON incidents(behaviour_type);
CREATE INDEX IF NOT EXISTS idx_incidents_bay ON incidents(bay);
CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);
