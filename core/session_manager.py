import sqlite3
import time
import json
from datetime import datetime
from pathlib import Path
import yaml

with open('config.yaml') as f:
    CFG = yaml.safe_load(f)

DB_PATH = Path(CFG['database']['path'])


def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time    TEXT NOT NULL,
            end_time      TEXT,
            duration_s    INTEGER DEFAULT 0,
            subject       TEXT DEFAULT "General",
            avg_focus     REAL DEFAULT 0,
            peak_focus    REAL DEFAULT 0,
            low_focus     REAL DEFAULT 100,
            distractions  INTEGER DEFAULT 0,
            yawns         INTEGER DEFAULT 0,
            blinks_total  INTEGER DEFAULT 0,
            face_lost_s   INTEGER DEFAULT 0,
            timeline_json TEXT DEFAULT "[]"
        )
    ''')
    con.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER REFERENCES sessions(id),
            timestamp  TEXT,
            event_type TEXT,
            detail     TEXT
        )
    ''')
    con.commit()
    return con


class SessionManager:

    def __init__(self, subject='General'):
        self.subject      = subject
        self.start_time   = datetime.now()
        self._db          = init_db()
        self._session_id  = None
        self.timeline     = []
        self.events       = []
        self.distractions = 0
        self.yawns        = 0
        self.blinks_total = 0
        self.face_lost_s  = 0
        self._prev_label  = 'Deep focus'
        self._no_face_ctr = 0

        cur = self._db.execute(
            'INSERT INTO sessions (start_time, subject) VALUES (?, ?)',
            (self.start_time.isoformat(), subject)
        )
        self._db.commit()
        self._session_id = cur.lastrowid

    def record(self, state, metrics):
        ts = time.time()
        self.timeline.append((ts, state.score))

        if metrics.blink_detected:
            self.blinks_total += 1

        if not metrics.face_visible:
            self._no_face_ctr += 1
            if self._no_face_ctr % 30 == 0:
                self.face_lost_s += 1
        else:
            self._no_face_ctr = 0

        if state.alert:
            self.events.append((ts, 'alert', state.alert))
            if 'fatigue' in state.alert.lower() or 'yawn' in state.alert.lower():
                self.yawns += 1
            elif 'away' in state.alert.lower():
                if self._prev_label not in ('Distracted', 'Disengaged'):
                    self.distractions += 1

        self._prev_label = state.label

    def save(self):
        end_time = datetime.now()
        duration = int((end_time - self.start_time).total_seconds())
        scores   = [s for _, s in self.timeline]

        avg  = round(sum(scores) / len(scores), 1) if scores else 0.0
        peak = round(max(scores), 1) if scores else 0.0
        low  = round(min(scores), 1) if scores else 0.0

        self._db.execute('''
            UPDATE sessions SET
                end_time=?, duration_s=?, avg_focus=?, peak_focus=?,
                low_focus=?, distractions=?, yawns=?, blinks_total=?,
                face_lost_s=?, timeline_json=?
            WHERE id=?
        ''', (
            end_time.isoformat(), duration, avg, peak, low,
            self.distractions, self.yawns, self.blinks_total,
            self.face_lost_s, json.dumps(self.timeline),
            self._session_id
        ))
        for ts, etype, detail in self.events:
            self._db.execute(
                'INSERT INTO events (session_id,timestamp,event_type,detail) '
                'VALUES (?,?,?,?)',
                (self._session_id, str(ts), etype, detail)
            )
        self._db.commit()

        return {
            'id':           self._session_id,
            'subject':      self.subject,
            'start_time':   self.start_time.isoformat(),
            'end_time':     end_time.isoformat(),
            'duration_s':   duration,
            'avg_focus':    avg,
            'peak_focus':   peak,
            'low_focus':    low,
            'distractions': self.distractions,
            'yawns':        self.yawns,
            'blinks_total': self.blinks_total,
            'timeline':     self.timeline
        }

    def get_past_sessions(self, limit=20):
        cur = self._db.execute('''
            SELECT id, start_time, subject, duration_s, avg_focus,
                   peak_focus, distractions, yawns, blinks_total
            FROM sessions WHERE end_time IS NOT NULL
            ORDER BY start_time DESC LIMIT ?
        ''', (limit,))
        return cur.fetchall()