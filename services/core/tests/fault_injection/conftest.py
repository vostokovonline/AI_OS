"""
Shared fixtures for P3 Fault Injection tests.
"""

import json
import os
import tempfile
import hashlib
from execution_dynamics.jsonl_wal import JsonLinesWAL
from execution_dynamics.journal import DispatchJournal, JournalEntry


def build_wal_with_events(path: str, events: list, goal_id: str = 'goal_1',
                          execution_id: str = 'exec1') -> JsonLinesWAL:
    """Create a WAL populated with the given event sequence via DispatchJournal.

    Routes through DispatchJournal.append() so hash chain is properly computed
    and stored in both the WAL file and JournalEntry objects.
    """
    wal = JsonLinesWAL(path)
    journal = DispatchJournal(wal=wal)
    for i, ev in enumerate(events):
        entry = JournalEntry(
            event=ev,
            goal_id=goal_id,
            execution_id=execution_id,
            lease_id=f'lease_{execution_id}',
            timestamp=1000.0 + i,
        )
        journal.append(entry)
    return wal


def jhash(entries) -> str:
    """Compute deterministic hash of a journal's full state."""
    raw = json.dumps([e.to_dict() for e in entries], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def state_hash_from_wal(path: str) -> str:
    """Replay WAL into DispatchJournal and hash the recovered state."""
    wal = JsonLinesWAL(path)
    journal = DispatchJournal(wal=wal)
    journal.recover_from_wal()
    return journal_hash(journal)


def journal_hash(journal: DispatchJournal) -> str:
    """Deterministic SHA256 of journal entries (sorted canonical)."""
    return jhash(journal._entries)
