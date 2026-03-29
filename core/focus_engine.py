import numpy as np
import yaml
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from core.face_tracker import FaceMetrics

with open('config.yaml') as f:
    CFG = yaml.safe_load(f)

W  = CFG['focus_engine']['weights']
FE = CFG['focus_engine']
TH = CFG['thresholds']
AL = CFG['alerts']


@dataclass
class FocusState:
    score:         float
    label:         str
    alert:         Optional[str]
    fatigue_level: float
    raw_score:     float
    signals:       dict


class FocusEngine:

    ALPHA  = FE['ema_alpha']
    WINDOW = FE['window_frames']

    def __init__(self):
        self._ema       = 75.0
        self._history   = deque(maxlen=self.WINDOW)
        self._alert_log = {}
        self._cooldown  = AL['cooldown_s']

    def _gaze_score(self, gx, gy):
        deviation = np.sqrt(gx**2 + gy**2)
        return float(max(0.0, 1.0 - deviation / TH['gaze_limit']) * 100)

    def _head_pose_score(self, pitch, yaw, roll):
        yaw_pen   = max(0.0, abs(yaw)   - TH['head_yaw_limit'])   / 30.0
        pitch_pen = max(0.0, abs(pitch) - TH['head_pitch_limit']) / 40.0
        roll_pen  = max(0.0, abs(roll)  - 15.0)                   / 30.0
        penalty   = min(1.0, yaw_pen + pitch_pen * 0.5 + roll_pen * 0.3)
        return float((1.0 - penalty) * 100)

    def _ear_score(self, ear):
        return float(min(100.0, max(0.0, (ear - 0.15) / 0.20 * 100)))

    def _blink_rate_score(self, bpm):
        if 12 <= bpm <= 20:
            return 100.0
        elif bpm < 12:
            return max(30.0, 100.0 - (12.0 - bpm) * 8.0)
        else:
            return max(20.0, 100.0 - (bpm - 20.0) * 5.0)

    def _yawn_score(self, yawning):
        return 0.0 if yawning else 100.0

    def _check_alert(self, m, bpm):
        now = time.time()

        def ok(key):
            return (now - self._alert_log.get(key, 0)) > self._cooldown

        if m.yawn_detected and ok('yawn'):
            self._alert_log['yawn'] = now
            return 'Fatigue detected — take a 5-minute break'
        if abs(m.head_yaw) > 35 and ok('yaw'):
            self._alert_log['yaw'] = now
            return 'You are looking away — refocus on your screen'
        if bpm < 8 and bpm > 0 and ok('blink'):
            self._alert_log['blink'] = now
            return 'Very low blink rate — rest your eyes'
        if m.eyes_closed and ok('closed'):
            self._alert_log['closed'] = now
            return 'Eyes closed — your session is still running!'
        if not m.face_visible and ok('noface'):
            self._alert_log['noface'] = now
            return 'Face not visible — are you still there?'
        return None

    def update(self, metrics, bpm):
        if not metrics.face_visible:
            signals = {k: 0.0 for k in W}
            raw = 0.0
        else:
            signals = {
                'gaze':       self._gaze_score(metrics.gaze_x, metrics.gaze_y),
                'head_pose':  self._head_pose_score(
                                  metrics.head_pitch, metrics.head_yaw, metrics.head_roll),
                'ear':        self._ear_score(metrics.ear),
                'blink_rate': self._blink_rate_score(bpm),
                'yawn':       self._yawn_score(metrics.yawn_detected),
            }
            raw = sum(signals[k] * W[k] for k in W)

        self._ema = self.ALPHA * raw + (1.0 - self.ALPHA) * self._ema
        self._history.append(self._ema)

        score = round(self._ema, 1)

        if   score >= 80: label = 'Deep focus'
        elif score >= 60: label = 'Moderate focus'
        elif score >= 40: label = 'Distracted'
        else:             label = 'Disengaged'

        fatigue = min(1.0, max(0.0, (bpm - 15.0) / 20.0))
        alert   = self._check_alert(metrics, bpm)

        return FocusState(
            score=score, label=label, alert=alert,
            fatigue_level=fatigue, raw_score=round(raw, 1),
            signals=signals
        )

    @property
    def session_average(self):
        return round(float(np.mean(self._history)), 1) if self._history else 0.0