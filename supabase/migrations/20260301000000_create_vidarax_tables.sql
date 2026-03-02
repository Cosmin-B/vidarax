-- Vidarax agent events table
CREATE TABLE agent_events (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  session_id TEXT NOT NULL DEFAULT '',
  frame_index BIGINT NOT NULL DEFAULT 0,
  pts_ms BIGINT NOT NULL DEFAULT 0,
  event_type TEXT NOT NULL,
  confidence FLOAT NOT NULL DEFAULT 0,
  description TEXT NOT NULL DEFAULT '',
  timestamp_ms BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_agent_events_run_id ON agent_events(run_id);
ALTER PUBLICATION supabase_realtime ADD TABLE agent_events;

-- Vidarax keyframe store table
CREATE TABLE keyframe_store (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  frame_index BIGINT NOT NULL DEFAULT 0,
  pts_ms BIGINT NOT NULL DEFAULT 0,
  event_type TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  jpeg_b64 TEXT NOT NULL DEFAULT '',
  timestamp_ms BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_keyframe_store_run_id ON keyframe_store(run_id);
ALTER PUBLICATION supabase_realtime ADD TABLE keyframe_store;
