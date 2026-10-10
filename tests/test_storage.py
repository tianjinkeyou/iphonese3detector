import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from phonewatch.storage import Store
from phonewatch.timer import UsageTimer


class SplitStorageTests(unittest.TestCase):
    def test_transition_interval_belongs_to_previous_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp))
            timer=UsageTimer(0)
            at=datetime(2026,10,10,12,0,0)
            for observation,now in [(False,0),(None,10),(True,30),(True,40)]:
                elapsed,changed=timer.update(observation,now)
                store.record(at,elapsed,timer.state,changed,timer.interval_state)
            self.assertEqual(store.totals('2026-10-10'),{'absent':10,'unknown':20,'legacy':0})
            store.close()

    def test_midnight_unknown_and_restart_persist_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp))
            store.record(datetime(2026,10,11,0,0,5),10,'unknown',True,'unknown')
            store.record(datetime(2026,10,11,0,0,15),10,'absent',True,'absent')
            store.close()
            store=Store(Path(tmp))
            self.assertEqual(store.totals('2026-10-10'),{'absent':0,'unknown':5,'legacy':0})
            self.assertEqual(store.totals('2026-10-11'),{'absent':10,'unknown':5,'legacy':0})
            store.close()

    def test_old_mixed_totals_are_preserved_without_guessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'usage.sqlite3'
            db=sqlite3.connect(path)
            db.execute('CREATE TABLE daily (day TEXT PRIMARY KEY, seconds REAL NOT NULL)')
            db.execute('INSERT INTO daily VALUES (?,?)',('2026-10-10',600))
            db.commit();db.close()
            for _ in range(2):
                store=Store(Path(tmp))
                self.assertEqual(store.totals('2026-10-10'),{'absent':0,'unknown':0,'legacy':600})
                store.close()
            store=Store(Path(tmp))
            store.record(datetime(2026,10,10,12,0,0),30,'unknown',True,'unknown')
            self.assertEqual(store.totals('2026-10-10'),{'absent':0,'unknown':30,'legacy':600})
            store.close()
