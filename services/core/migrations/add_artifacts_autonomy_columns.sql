-- Add Autonomy v2 columns to artifacts table
-- Phase 2.5.P: State Mutations and Decision Signals
-- Date: 2026-02-21

-- These columns enable artifacts to propose system state changes
-- and provide decision signals for the autonomous system

-- Add state_mutations column
-- Stores proposed state changes: [{"entity_name": "...", "mutation_type": "update", "new_value": {...}}]
ALTER TABLE artifacts
ADD COLUMN IF NOT EXISTS state_mutations JSON;

-- Add decision_signals column
-- Stores decision signals: [{"signal_type": "positive", "strength": 0.7}]
ALTER TABLE artifacts
ADD COLUMN IF NOT EXISTS decision_signals JSON;

-- Verification query
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'artifacts'
  AND column_name IN ('state_mutations', 'decision_signals')
ORDER BY column_name;
