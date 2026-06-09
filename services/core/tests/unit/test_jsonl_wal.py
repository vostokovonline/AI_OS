"""
JSON Lines WAL Tests — P2.7 Execution WAL persistence.

Covers:
  - Basic append + replay
  - Fsync durability (simulated crash)
  - Crash recovery (truncate last invalid JSON line)
  - Sequence consistency
  - Hash chain at WAL level
  - Empty / missing file
  - Multiple writers (sequential)
  - Concurrent safety (basic)
"""

import json
import os
import tempfile
import time
from execution_dynamics.jsonl_wal import JsonLinesWAL, WalEntry


def make_payload(event: str, goal_id: str = "g1", execution_id: str = "e1",
                 entry_hash: str = "abc123") -> dict:
    return {
        'event': event,
        'goal_id': goal_id,
        'execution_id': execution_id,
        'entry_hash': entry_hash,
    }


class TestAppendAndReplay:

    def test_append_single_entry(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            lsn = wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED'))
            assert lsn != ""
            assert wal.get_entry_count() == 1

            entries = wal.replay()
            assert len(entries) == 1
            assert entries[0].entry_type == 'DISPATCHED'
            assert entries[0].entry_id == 'e1:0:dispatched'
        finally:
            os.unlink(path)

    def test_append_multiple_entries_preserves_order(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            events = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
            for i, ev in enumerate(events):
                eid = f"e1:{i}:{ev.lower()}"
                wal.append(ev, eid, make_payload(ev, entry_hash=f"hash_{i}"))

            assert wal.get_entry_count() == 4
            entries = wal.replay()
            assert [e.entry_type for e in entries] == events
        finally:
            os.unlink(path)

    def test_replay_after_lsn(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            lsns = []
            for i, ev in enumerate(['DISPATCHED', 'STARTED', 'COMPLETED']):
                eid = f"e1:{i}:{ev.lower()}"
                lsn = wal.append(ev, eid, make_payload(ev, entry_hash=f"hash_{i}"))
                lsns.append(lsn)

            # Replay after first LSN → should get entries 2 and 3
            entries = wal.replay(since_lsn=lsns[0])
            assert len(entries) == 2
            assert entries[0].entry_type == 'STARTED'
            assert entries[1].entry_type == 'COMPLETED'
        finally:
            os.unlink(path)


class TestFsyncDurability:

    def test_entries_survive_reopen(self):
        """Simulate crash: write, close, reopen — entries must be present."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED'))
            wal.append('COMPLETED', 'e1:1:completed', make_payload('COMPLETED'))
            del wal  # Close

            # Reopen — simulate restart after crash
            wal2 = JsonLinesWAL(path)
            assert wal2.get_entry_count() == 2
            entries = wal2.replay()
            assert entries[0].entry_type == 'DISPATCHED'
            assert entries[1].entry_type == 'COMPLETED'
        finally:
            os.unlink(path)

    def test_reopen_append_cold_start(self):
        """Cold start: file exists with N entries, append N+1."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED'))
            del wal

            wal2 = JsonLinesWAL(path)
            wal2.append('STARTED', 'e1:1:started', make_payload('STARTED'))
            entries = wal2.replay()
            assert len(entries) == 2
        finally:
            os.unlink(path)


class TestCrashRecovery:

    def test_truncate_partial_last_line(self):
        """Simulate crash mid-write: last line is incomplete JSON → truncate."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED', entry_hash='h1'))
            wal.append('STARTED', 'e1:1:started', make_payload('STARTED', entry_hash='h2'))
            del wal

            # Corrupt last line: append incomplete JSON
            with open(path, 'a') as f:
                f.write('{"lsn": "corrupted", "entry_type": "PAR')
                f.flush()
                os.fsync(f.fileno())

            # Recover — should truncate the incomplete line
            wal2 = JsonLinesWAL(path)
            assert wal2.get_entry_count() == 2
            entries = wal2.replay()
            assert [e.entry_type for e in entries] == ['DISPATCHED', 'STARTED']
        finally:
            os.unlink(path)

    def test_truncate_after_binary_garbage(self):
        """Binary garbage injected → truncate at garbage boundary."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED'))
            del wal

            # Append garbage
            with open(path, 'ab') as f:
                f.write(b'\x00\x01\x02\xff\xfe\xfd')
                f.flush()
                os.fsync(f.fileno())

            wal2 = JsonLinesWAL(path)
            assert wal2.get_entry_count() == 1
        finally:
            os.unlink(path)

    def test_truncate_multiple_corrupt_lines(self):
        """Multiple corrupt lines after valid prefix → all truncated."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED'))
            del wal

            with open(path, 'a') as f:
                f.write('NOT JSON\n')
                f.write('ALSO NOT JSON\n')
                f.flush()
                os.fsync(f.fileno())

            wal2 = JsonLinesWAL(path)
            assert wal2.get_entry_count() == 1
        finally:
            os.unlink(path)

    def test_empty_file_on_open(self):
        """Empty file — no entries, no crash."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            assert wal.get_entry_count() == 0
            entries = wal.replay()
            assert entries == []
        finally:
            os.unlink(path)

    def test_missing_file_creates_on_append(self):
        """File doesn't exist → created on first append."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wal.jsonl")
            assert not os.path.exists(path)

            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED'))

            assert os.path.exists(path)
            assert wal.get_entry_count() == 1


class TestSequence:

    def test_monotonic_sequence(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            for i in range(10):
                wal.append('DISPATCHED', f"e1:{i}:dispatched", make_payload('DISPATCHED'))

            entries = wal.replay()
            seqs = [e.seq for e in entries]
            assert seqs == list(range(10))
        finally:
            os.unlink(path)

    def test_sequence_survives_reopen(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            for i in range(5):
                wal.append('DISPATCHED', f"e1:{i}:dispatched", make_payload('DISPATCHED'))
            del wal

            wal2 = JsonLinesWAL(path)
            wal2.append('STARTED', 'e1:5:started', make_payload('STARTED'))
            entries = wal2.replay()
            assert entries[5].seq == 5
        finally:
            os.unlink(path)


class TestHashChain:

    def test_prev_hash_links_consecutive_entries(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            hashes = ['h0', 'h1', 'h2']
            for i, h in enumerate(hashes):
                wal.append('DISPATCHED', f"e1:{i}:dispatched",
                           make_payload('DISPATCHED', entry_hash=h))

            entries = wal.replay()
            assert entries[0].prev_hash == ""
            assert entries[1].prev_hash == 'h0'
            assert entries[2].prev_hash == 'h1'
        finally:
            os.unlink(path)

    def test_prev_hash_survives_reopen(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched',
                       make_payload('DISPATCHED', entry_hash='h0'))
            wal.append('STARTED', 'e1:1:started',
                       make_payload('STARTED', entry_hash='h1'))
            del wal

            wal2 = JsonLinesWAL(path)
            entries = wal2.replay()
            assert entries[0].prev_hash == ""
            assert entries[1].prev_hash == 'h0'
        finally:
            os.unlink(path)

    def test_entry_hash_persisted(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched',
                       make_payload('DISPATCHED', entry_hash='my_test_hash'))
            del wal

            wal2 = JsonLinesWAL(path)
            entries = wal2.replay()
            assert entries[0].entry_hash == 'my_test_hash'
        finally:
            os.unlink(path)


class TestLSN:

    def test_lsn_monotonic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            lsns = []
            for i in range(5):
                lsn = wal.append('DISPATCHED', f"e1:{i}:dispatched", make_payload('DISPATCHED'))
                lsns.append(lsn)

            # LSNs must be unique
            assert len(set(lsns)) == 5
            # Each LSN's sequence number (last component) must be strictly increasing
            seqs = [int(lsn.split('-')[-1]) for lsn in lsns]
            assert seqs == sorted(seqs)
            assert seqs == list(range(5))
        finally:
            os.unlink(path)


class TestStats:

    def test_stats_empty(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            stats = wal.get_stats()
            assert stats['n_entries'] == 0
            assert stats['file_size_bytes'] >= 0
        finally:
            os.unlink(path)

    def test_stats_after_append(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            wal.append('DISPATCHED', 'e1:0:dispatched', make_payload('DISPATCHED'))
            stats = wal.get_stats()
            assert stats['n_entries'] == 1
            assert stats['file_size_bytes'] > 0
            assert stats['path'] == path
        finally:
            os.unlink(path)


class TestIntegrationWithDispatchJournal:

    def test_journal_append_with_jsonl_wal(self):
        """DispatchJournal with JsonLinesWAL: append → replay → reconstruct state."""
        from execution_dynamics.journal import DispatchJournal, JournalEntry

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            journal = DispatchJournal(wal=wal)

            entry = JournalEntry(
                event='DISPATCHED',
                goal_id='goal_1',
                execution_id='exec_1',
                lease_id='lease_1',
                timestamp=1000.0,
            )
            journal.append(entry)
            assert journal.get_stats()['total_entries'] == 1
            assert wal.get_entry_count() == 1
        finally:
            os.unlink(path)

    def test_journal_recover_from_jsonl_wal(self):
        """Simulate crash: write via journal, reopen, recover from WAL."""
        from execution_dynamics.journal import DispatchJournal, JournalEntry

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            # First session
            wal = JsonLinesWAL(path)
            journal = DispatchJournal(wal=wal)

            for ev in ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']:
                journal.append(JournalEntry(
                    event=ev,
                    goal_id='goal_1',
                    execution_id='exec_1',
                    lease_id='lease_1',
                    timestamp=1000.0,
                ))

            assert journal.get_stats()['total_entries'] == 4
            del journal
            del wal

            # Crash recovery: reopen WAL + journal
            wal2 = JsonLinesWAL(path)
            journal2 = DispatchJournal(wal=wal2)
            recovered = journal2.recover_from_wal()
            assert recovered == 4

            # Verify state matches
            chain = journal2.get_chain('goal_1')
            assert len(chain) == 4
            assert [e.event for e in chain] == ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
            assert chain[-1].event == 'COMPLETED'
        finally:
            os.unlink(path)

    def test_journal_recover_after_truncation(self):
        """Crash mid-write → truncate → recover → consistent state."""
        from execution_dynamics.journal import DispatchJournal, JournalEntry

        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = JsonLinesWAL(path)
            journal = DispatchJournal(wal=wal)

            journal.append(JournalEntry(event='DISPATCHED', goal_id='g1',
                                        execution_id='e1', lease_id='l1', timestamp=1000.0))
            journal.append(JournalEntry(event='STARTED', goal_id='g1',
                                        execution_id='e1', lease_id='l1', timestamp=1001.0))
            del journal
            del wal

            # Corrupt: append partial JSON line
            with open(path, 'a') as f:
                f.write('{"lsn": "broken", "entry_type": "COM')
                f.flush()
                os.fsync(f.fileno())

            # Recover — should truncate partial line
            wal2 = JsonLinesWAL(path)
            journal2 = DispatchJournal(wal=wal2)
            recovered = journal2.recover_from_wal()
            assert recovered == 2

            chain = journal2.get_chain('g1')
            assert len(chain) == 2
            assert chain[-1].event == 'STARTED'
        finally:
            os.unlink(path)
