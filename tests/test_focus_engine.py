import pytest
from core.face_tracker import FaceMetrics
from core.focus_engine import FocusEngine


def make_metrics(**kwargs):
    defaults = dict(
        ear_left=0.30, ear_right=0.30, ear=0.30,
        mar=0.20, gaze_x=0.0, gaze_y=0.0,
        head_pitch=5.0, head_yaw=5.0, head_roll=2.0,
        blink_detected=False, yawn_detected=False,
        eyes_closed=False, face_visible=True
    )
    defaults.update(kwargs)
    return FaceMetrics(**defaults)


@pytest.fixture
def engine():
    return FocusEngine()


def test_good_conditions_score(engine):
    for _ in range(60):
        state = engine.update(make_metrics(), bpm=15)
    assert state.score > 55


def test_large_yaw_lowers_score(engine):
    for _ in range(30):
        engine.update(make_metrics(), bpm=15)
    base = engine.session_average
    for _ in range(40):
        engine.update(make_metrics(head_yaw=55), bpm=15)
    assert engine.session_average < base


def test_yawn_triggers_alert(engine):
    state = engine.update(
        make_metrics(yawn_detected=True, mar=0.8), bpm=15
    )
    assert state.alert is not None


def test_no_face_zeroes_score(engine):
    for _ in range(120):
        engine.update(FaceMetrics(face_visible=False), bpm=0)
    assert engine.session_average < 25


def test_ema_smoothing(engine):
    for _ in range(40):
        engine.update(make_metrics(), bpm=15)
    avg_before = engine.session_average
    engine.update(make_metrics(head_yaw=80), bpm=30)
    assert engine.session_average > avg_before - 6