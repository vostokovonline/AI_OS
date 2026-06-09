-- Adds last_activity_at column to goals table for GoalState tracking.
-- Run against existing databases that were created before this column existed.

ALTER TABLE goals ADD COLUMN IF NOT EXISTS last_activity_at TIMESTAMP WITH TIME ZONE;
