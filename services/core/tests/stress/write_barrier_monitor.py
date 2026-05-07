"""
Write Barrier Monitor
"""
from typing import List

class WriteBarrierMonitor:
    def __init__(self):
        self.write_phase_started = False
        self.violations = []

    def allow_writes(self):
        self.write_phase_started = True

    def before_execute(self, conn, cursor, statement, parameters, context, executemany):
        sql = statement.strip().lower() if statement else ""
        is_write = sql.startswith(('insert', 'update', 'delete'))
        if is_write and not self.write_phase_started:
            self.violations.append(sql)
            raise AssertionError(f'WRITE before barrier: {sql[:50]}')
