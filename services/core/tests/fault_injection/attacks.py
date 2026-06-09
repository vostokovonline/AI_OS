"""
WAL Attack Primitives — P3 Fault Injection Framework.

4 categories of attacks corresponding to 4 invariants:
  A. Physical Corruption  → tests JsonLinesWAL recovery
  B. Structural Corruption → tests IntegrityVerifier (sequence)
  C. Cryptographic Corruption → tests IntegrityVerifier (hash chain)
  D. Semantic Corruption   → tests IntegrityVerifier (lifecycle)
"""

import hashlib
import json
import os
import random
from typing import List, Dict, Any, Optional


# ============================================================================
# A. Physical Corruption — damages the WAL file bytes on disk
# ============================================================================

def truncate_last_byte(path: str):
    """Remove the last byte of the file. Simulates incomplete fsync."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    size = os.path.getsize(path)
    with open(path, 'wb') as f:
        f.truncate(size - 1)


def truncate_last_line(path: str):
    """Remove the last complete line. Simulates partial write."""
    if not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        lines = f.readlines()
    if len(lines) <= 1:
        # Not enough lines to truncate meaningfully
        return
    with open(path, 'wb') as f:
        for line in lines[:-1]:
            f.write(line)


def insert_binary_garbage_tail(path: str):
    """Append binary non-UTF8 bytes at end of file."""
    with open(path, 'ab') as f:
        f.write(b'\x00\x01\x02\xff\xfe\xfd')
        f.flush()
        os.fsync(f.fileno())


def insert_binary_garbage_middle(path: str):
    """Inject binary non-UTF8 bytes in the middle of the file."""
    if not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        content = f.read()
    split = max(1, len(content) * 3 // 5)
    content = content[:split] + b'\xde\xad\xbe\xef' + content[split:]
    with open(path, 'wb') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def partial_json_tail(path: str):
    """Append incomplete JSON at end of file."""
    with open(path, 'ab') as f:
        f.write(b'{"entry_type": "PA')
        f.flush()
        os.fsync(f.fileno())


def partial_json_middle(path: str):
    """Inject incomplete JSON in middle of valid content."""
    if not os.path.exists(path):
        return
    with open(path, 'rb') as f:
        content = f.read()
    split = max(1, len(content) * 2 // 3)
    content = content[:split] + b'{"entry_type": "IN' + content[split:]
    with open(path, 'wb') as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


PHYSICAL_ATTACKS = [
    truncate_last_byte,
    truncate_last_line,
    insert_binary_garbage_tail,
    insert_binary_garbage_middle,
    partial_json_tail,
    partial_json_middle,
]


# ============================================================================
# B. Structural Corruption — breaks journal structure (valid JSON, bad semantics)
# ============================================================================

def _read_entries(path: str) -> List[dict]:
    """Read JSON lines from WAL file, return list of dicts."""
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                try:
                    entries.append(json.loads(stripped))
                except json.JSONDecodeError:
                    break
    return entries


def _write_entries(path: str, entries: List[dict]):
    """Write list of dicts as JSON lines to WAL file."""
    with open(path, 'wb') as f:
        for entry in entries:
            f.write((json.dumps(entry, sort_keys=True) + '\n').encode('utf-8'))
        f.flush()
        os.fsync(f.fileno())


def delete_middle_entry(path: str):
    """Remove an entry from the middle of the WAL."""
    entries = _read_entries(path)
    if len(entries) < 3:
        return
    mid = len(entries) // 2
    entries.pop(mid)
    # Rebuild file
    _write_entries(path, entries)


def duplicate_entry(path: str):
    """Duplicate the last entry at the end of the WAL."""
    entries = _read_entries(path)
    if not entries:
        return
    entries.append(dict(entries[-1]))
    _write_entries(path, entries)


def swap_entries(path: str):
    """Swap two entries in the middle of the WAL."""
    entries = _read_entries(path)
    if len(entries) < 4:
        return
    i, j = 1, 2
    entries[i], entries[j] = entries[j], entries[i]
    _write_entries(path, entries)


def reorder_range(path: str):
    """Reverse a range of entries (3 entries) in the middle."""
    entries = _read_entries(path)
    if len(entries) < 5:
        return
    start = max(1, len(entries) // 2 - 1)
    end = min(start + 3, len(entries))
    entries[start:end] = reversed(entries[start:end])
    _write_entries(path, entries)


def corrupt_sequence(path: str):
    """Set entry_id of a middle entry to an out-of-order sequence index."""
    entries = _read_entries(path)
    if len(entries) < 3:
        return
    mid = len(entries) // 2
    # Change the sequence index in entry_id (format: "exec:seq:event")
    old_id = entries[mid].get('entry_id', '')
    parts = old_id.rsplit(':', 2)
    if len(parts) == 3:
        parts[1] = '9999'
        new_id = ':'.join(parts)
        entries[mid]['entry_id'] = new_id
        if 'payload' in entries[mid] and isinstance(entries[mid]['payload'], dict):
            entries[mid]['payload']['entry_id'] = new_id
    _write_entries(path, entries)


STRUCTURAL_ATTACKS = [
    delete_middle_entry,
    duplicate_entry,
    swap_entries,
    reorder_range,
    corrupt_sequence,
]


# ============================================================================
# C. Cryptographic Corruption — breaks hash chain integrity
# ============================================================================

def corrupt_entry_hash(path: str):
    """Modify goal_id in a middle entry (changes hash without affecting event type)."""
    entries = _read_entries(path)
    if len(entries) < 3:
        return
    mid = len(entries) // 2
    if 'payload' in entries[mid] and isinstance(entries[mid]['payload'], dict):
        entries[mid]['payload']['goal_id'] = 'corrupted_goal'
    _write_entries(path, entries)


def corrupt_prev_hash(path: str):
    """Corrupt prev_hash at the outer level and in payload."""
    entries = _read_entries(path)
    if len(entries) < 2:
        return
    target = random.randint(1, len(entries) - 1)
    for entry in [entries[target]]:
        entry['prev_hash'] = 'broken_chain'
        if 'payload' in entry and isinstance(entry['payload'], dict):
            entry['payload']['prev_hash'] = 'broken_chain'
    _write_entries(path, entries)


def break_hash_chain(path: str):
    """Replace all entry_hash and prev_hash with zero hashes (outer + payload)."""
    entries = _read_entries(path)
    if not entries:
        return
    ZERO_HASH = '0' * 64
    for entry in entries:
        entry['entry_hash'] = ZERO_HASH
        entry['prev_hash'] = ZERO_HASH
        if 'payload' in entry and isinstance(entry['payload'], dict):
            entry['payload']['entry_hash'] = ZERO_HASH
            entry['payload']['prev_hash'] = ZERO_HASH
    _write_entries(path, entries)


def forge_entry_payload(path: str):
    """Modify event_type in a middle entry (makes hash invalid)."""
    entries = _read_entries(path)
    if len(entries) < 3:
        return
    mid = len(entries) // 2
    if 'payload' in entries[mid] and isinstance(entries[mid]['payload'], dict):
        entries[mid]['payload']['event'] = 'FORGED'
    _write_entries(path, entries)


CRYPTOGRAPHIC_ATTACKS = [
    corrupt_entry_hash,
    corrupt_prev_hash,
    break_hash_chain,
    forge_entry_payload,
]


# ============================================================================
# D. Semantic Corruption — breaks state machine (invalid lifecycle transitions)
# ============================================================================

def replace_event(path: str, target_idx: int, new_event: str):
    """Replace event_type at target_idx with new_event."""
    entries = _read_entries(path)
    if target_idx >= len(entries):
        return
    entries[target_idx]['entry_type'] = new_event
    if 'payload' in entries[target_idx] and isinstance(entries[target_idx]['payload'], dict):
        entries[target_idx]['payload']['event'] = new_event
    _write_entries(path, entries)


def inject_duplicate_started(path: str):
    """Replace the final entry with another STARTED (COMPLETED→STARTED)."""
    entries = _read_entries(path)
    if not entries:
        return
    entries[-1]['entry_type'] = 'STARTED'
    if 'payload' in entries[-1] and isinstance(entries[-1]['payload'], dict):
        entries[-1]['payload']['event'] = 'STARTED'
    _write_entries(path, entries)


def inject_cycle(path: str):
    """Replace a middle entry to create DISPATCHED→COMPLETED→DISPATCHED loop."""
    entries = _read_entries(path)
    if len(entries) < 4:
        return
    entries[-2]['entry_type'] = 'DISPATCHED'
    if 'payload' in entries[-2] and isinstance(entries[-2]['payload'], dict):
        entries[-2]['payload']['event'] = 'DISPATCHED'
    _write_entries(path, entries)


def inject_orphan_started(path: str):
    """Remove first entry (DISPATCHED) to leave orphan STARTED."""
    entries = _read_entries(path)
    if len(entries) < 2:
        return
    entries[0]['entry_type'] = 'STARTED'
    if 'payload' in entries[0] and isinstance(entries[0]['payload'], dict):
        entries[0]['payload']['event'] = 'STARTED'
    _write_entries(path, entries)


SEMANTIC_ATTACKS = [
    inject_duplicate_started,
    inject_cycle,
    inject_orphan_started,
]


ALL_ATTACKS = PHYSICAL_ATTACKS + STRUCTURAL_ATTACKS + CRYPTOGRAPHIC_ATTACKS + SEMANTIC_ATTACKS
