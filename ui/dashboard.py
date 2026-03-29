import streamlit as st
import cv2
import time
import yaml
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from core.face_tracker    import FaceTracker
from core.focus_engine    import FocusEngine
from core.session_manager import SessionManager, init_db

with open('config.yaml') as f:
    CFG = yaml.safe_load(f)

st.set_page_config(
    page_title='NeuroPose',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title('🧠 NeuroPose')
    st.caption('Cognitive attention monitoring via facial analysis')
    st.divider()

    st.subheader('Session Settings')
    subject   = st.text_input('Subject / Topic', 'Mathematics')
    cam_index = st.number_input('Camera index (try 0, 1, or 2)', 0, 4, 0)
    pomodoro  = st.slider('Pomodoro duration (minutes)', 5, 60, 25)

    st.divider()
    st.subheader('Display Options')
    show_overlay = st.toggle('Show face overlay on video', True)
    show_signals = st.toggle('Show signal breakdown', True)

    st.divider()
    st.markdown('**How focus is measured:**')
    st.caption('• Gaze direction (iris tracking)')
    st.caption('• Head orientation (yaw/pitch/roll)')
    st.caption('• Eye openness (EAR formula)')
    st.caption('• Blink rate (12–20/min = healthy)')
    st.caption('• Yawn detection (MAR formula)')

# ── Tabs ─────────────────────────────────────────────────────────────
tab_live, tab_history, tab_about = st.tabs([
    '📹 Live Session',
    '📊 Session History',
    'ℹ️ How It Works'
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SESSION
# ════════════════════════════════════════════════════════════════════
with tab_live:

    cam_col, stats_col = st.columns([1.6, 1])

    with cam_col:
        st.subheader('Live Feed')
        frame_slot = st.empty()
        alert_slot = st.empty()

    with stats_col:
        st.subheader('Focus Score')
        gauge_slot  = st.empty()
        signal_slot = st.empty()

    btn_col1, btn_col2 = st.columns(2)
    start_btn = btn_col1.button(
        '▶  Start Session',
        type='primary',
        use_container_width=True
    )
    stop_btn = btn_col2.button(
        '⏹  Stop & Save',
        use_container_width=True
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    avg_slot  = m1.empty()
    dist_slot = m2.empty()
    yawn_slot = m3.empty()
    bpm_slot  = m4.empty()
    time_slot = m5.empty()

    st.markdown('**Focus timeline**')
    timeline_slot = st.empty()

    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'session' not in st.session_state:
        st.session_state.session = None

    if start_btn:
        st.session_state.running = True
        st.session_state.session = SessionManager(subject)

    if stop_btn:
        st.session_state.running = False

    if st.session_state.running and st.session_state.session:
        tracker = FaceTracker()
        engine  = FocusEngine()
        session = st.session_state.session
        cap     = cv2.VideoCapture(int(cam_index))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            st.error(
                f'Cannot open camera {cam_index}. '
                'Try changing Camera index in the sidebar to 1 or 2.'
            )
        else:
            session_start = time.time()
            pomo_end      = session_start + pomodoro * 60

            while st.session_state.running:
                ok, frame = cap.read()
                if not ok:
                    st.warning('Camera read failed. Check your connection.')
                    break

                metrics = tracker.process(frame)
                bpm     = tracker.blinks_per_minute()
                state   = engine.update(metrics, bpm)
                session.record(state, metrics)

                # Coloured border based on focus level
                if   state.score >= 70: border_color = (0, 200, 80)
                elif state.score >= 45: border_color = (0, 165, 255)
                else:                   border_color = (0, 50,  220)

                h_f, w_f = frame.shape[:2]
                cv2.rectangle(
                    frame, (0, 0), (w_f - 1, h_f - 1),
                    border_color, 5
                )
                cv2.putText(
                    frame,
                    f'{state.score:.0f}%  {state.label}',
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, border_color, 2
                )

                if show_overlay:
                    tracker.draw_overlay(frame, metrics)

                frame_slot.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels='RGB',
                    use_column_width=True
                )

                if state.alert:
                    alert_slot.warning(f'⚠️ {state.alert}')
                else:
                    alert_slot.empty()

                # ── Gauge chart ────────────────────────────────────
                gauge_color = (
                    '#22c55e' if state.score >= 70 else
                    '#f59e0b' if state.score >= 45 else
                    '#ef4444'
                )
                fig_gauge = go.Figure(go.Indicator(
                    mode='gauge+number+delta',
                    value=state.score,
                    delta={
                        'reference': engine.session_average,
                        'relative': False,
                        'valueformat': '.1f'
                    },
                    title={'text': state.label, 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar':  {'color': gauge_color, 'thickness': 0.28},
                        'steps': [
                            {'range': [0,  40], 'color': '#fef2f2'},
                            {'range': [40, 70], 'color': '#fefce8'},
                            {'range': [70,100], 'color': '#f0fdf4'},
                        ],
                    },
                    number={'suffix': '%', 'font': {'size': 42}}
                ))
                fig_gauge.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=60, b=0),
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                gauge_slot.plotly_chart(fig_gauge, use_container_width=True)

                # ── Signal breakdown bars ──────────────────────────
                if show_signals and state.signals:
                    html_bars = ''
                    for sig_name, sig_val in state.signals.items():
                        bar_color = (
                            '#22c55e' if sig_val >= 70 else
                            '#f59e0b' if sig_val >= 40 else
                            '#ef4444'
                        )
                        html_bars += (
                            f'<div style="display:flex;align-items:center;'
                            f'gap:8px;margin:4px 0;font-size:12px">'
                            f'<span style="width:85px;color:#555">'
                            f'{sig_name}</span>'
                            f'<div style="flex:1;background:#e5e7eb;height:8px;'
                            f'border-radius:4px;overflow:hidden">'
                            f'<div style="width:{sig_val:.0f}%;'
                            f'background:{bar_color};height:100%">'
                            f'</div></div>'
                            f'<span style="width:38px;color:{bar_color};'
                            f'font-weight:600;text-align:right">'
                            f'{sig_val:.0f}%</span></div>'
                        )
                    signal_slot.markdown(html_bars, unsafe_allow_html=True)

                # ── Metrics row ────────────────────────────────────
                elapsed = int(time.time() - session_start)
                hrs  = elapsed // 3600
                mins = (elapsed % 3600) // 60
                secs = elapsed % 60
                avg_slot.metric('Avg focus',    f'{engine.session_average:.1f}%')
                dist_slot.metric('Distractions', session.distractions)
                yawn_slot.metric('Yawns',        session.yawns)
                bpm_slot.metric('Blinks/min',    f'{bpm:.0f}')
                time_slot.metric('Time',         f'{hrs:02d}:{mins:02d}:{secs:02d}')

                # ── Timeline chart ─────────────────────────────────
                if len(session.timeline) > 10:
                    _, scores_list = zip(*session.timeline[-600:])
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(
                        y=list(scores_list),
                        mode='lines',
                        line=dict(color='#6366f1', width=1.5),
                        fill='tozeroy',
                        fillcolor='rgba(99,102,241,0.07)'
                    ))
                    fig_t.add_hline(
                        y=70,
                        line_color='#22c55e',
                        line_dash='dot',
                        line_width=1
                    )
                    fig_t.update_layout(
                        height=120,
                        margin=dict(l=0, r=0, t=4, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(
                            range=[0, 100],
                            showgrid=False,
                            showticklabels=True,
                            tickfont={'size': 9}
                        ),
                        xaxis=dict(
                            showgrid=False,
                            showticklabels=False
                        ),
                        showlegend=False
                    )
                    timeline_slot.plotly_chart(
                        fig_t, use_container_width=True
                    )

                # ── Pomodoro check ─────────────────────────────────
                if time.time() >= pomo_end:
                    st.balloons()
                    st.success(
                        f'Pomodoro complete! '
                        f'Take a {CFG["pomodoro"]["break_min"]}-minute break.'
                    )
                    st.session_state.running = False
                    break

                time.sleep(0.033)

            cap.release()

            if st.session_state.session:
                summary = st.session_state.session.save()
                st.success(
                    f'Session saved!  |  '
                    f'Avg focus: {summary["avg_focus"]}%  |  '
                    f'Duration: {summary["duration_s"] // 60}m '
                    f'{summary["duration_s"] % 60}s'
                )
                # Generate PDF report
                try:
                    from analytics.report_generator import generate_report
                    import os
                    pdf_path = f'data/report_{summary["id"]}.pdf'
                    generate_report(summary, pdf_path)
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            label='📄 Download PDF Report',
                            data=f,
                            file_name=(
                                f'neuropose_report_'
                                f'{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
                            ),
                            mime='application/pdf'
                        )
                except Exception as e:
                    st.warning(f'PDF generation skipped: {e}')

# ════════════════════════════════════════════════════════════════════
# TAB 2 — SESSION HISTORY
# ════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader('Your Past Sessions')

    con  = init_db()
    rows = con.execute('''
        SELECT start_time, subject, duration_s, avg_focus,
               peak_focus, distractions, yawns, blinks_total
        FROM sessions
        WHERE end_time IS NOT NULL
        ORDER BY start_time DESC
        LIMIT 30
    ''').fetchall()

    if not rows:
        st.info(
            'No sessions recorded yet. '
            'Start a session in the Live tab to see history here.'
        )
    else:
        df = pd.DataFrame(rows, columns=[
            'Start', 'Subject', 'Duration_s', 'Avg Focus',
            'Peak Focus', 'Distractions', 'Yawns', 'Blinks'
        ])
        df['Duration'] = df['Duration_s'].apply(
            lambda s: f'{int(s) // 60}m {int(s) % 60}s'
        )
        df['Date'] = pd.to_datetime(
            df['Start']
        ).dt.strftime('%d %b %Y  %H:%M')

        k1, k2, k3, k4 = st.columns(4)
        k1.metric('Total sessions',    len(df))
        k2.metric('Overall avg focus', f'{df["Avg Focus"].mean():.1f}%')
        k3.metric('Total distractions', int(df['Distractions'].sum()))
        total_s = int(df['Duration_s'].sum())
        k4.metric(
            'Total study time',
            f'{total_s // 3600}h {(total_s % 3600) // 60}m'
        )

        st.divider()

        st.dataframe(
            df[[
                'Date', 'Subject', 'Duration', 'Avg Focus',
                'Peak Focus', 'Distractions', 'Yawns', 'Blinks'
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        fig_trend = px.line(
            df[::-1].reset_index(drop=True),
            y='Avg Focus',
            title='Focus trend across sessions',
            markers=True,
            labels={
                'Avg Focus': 'Avg focus (%)',
                'index': 'Session number'
            }
        )
        fig_trend.add_hline(
            y=70,
            line_dash='dot',
            line_color='green',
            annotation_text='Target (70%)'
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            height=300
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        if df['Subject'].nunique() > 1:
            subj_avg = (
                df.groupby('Subject')['Avg Focus']
                .mean()
                .reset_index()
            )
            fig_subj = px.bar(
                subj_avg,
                x='Subject',
                y='Avg Focus',
                title='Average focus by subject',
                color='Avg Focus',
                color_continuous_scale='RdYlGn',
                labels={'Avg Focus': 'Avg focus (%)'}
            )
            fig_subj.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                height=300
            )
            st.plotly_chart(fig_subj, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader('How NeuroPose Calculates Your Focus Score')

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('### Signal weights')
        st.markdown('''
| Signal | Weight | What it detects |
|--------|--------|-----------------|
| Gaze | 35% | Looking at screen vs away |
| Head pose | 25% | Head turned sideways |
| Eye openness | 20% | Eyes open vs closed |
| Blink rate | 12% | 12–20/min = healthy |
| Yawn | 8% | Mouth open = fatigue |
        ''')

    with col_right:
        st.markdown('### Score labels')
        st.success('80–100  →  Deep focus')
        st.info('60–79  →  Moderate focus')
        st.warning('40–59  →  Distracted')
        st.error('0–39  →  Disengaged')

    st.divider()

    st.markdown('### Technology stack')
    st.markdown('''
- **MediaPipe Face Mesh** — 468 3D landmarks on face at 30 fps
- **Iris landmarks** — 5 extra points per iris for gaze direction
- **EAR formula** — Eye Aspect Ratio detects blinks and closure
- **MAR formula** — Mouth Aspect Ratio detects yawns
- **OpenCV solvePnP** — solves head rotation from 3D to 2D points
- **EMA smoothing** — prevents score from jumping every frame
- **SQLite** — local database, no setup needed
- **Streamlit + Plotly** — live dashboard with interactive charts
    ''')

    st.divider()

    st.markdown('### Alert thresholds')
    st.code('''
Head yaw  > 35 degrees  →  "You are looking away"
Blink rate < 8 per min  →  "Rest your eyes"
Yawn detected           →  "Take a 5-minute break"
Eyes closed > 0.5 sec   →  "Eyes closed — wake up!"
Face not visible        →  "Are you still there?"
    ''')

    st.divider()

    st.markdown('### About NeuroPose')
    st.markdown('''
NeuroPose is a real-time cognitive attention monitoring system built
as a portfolio project for IIT Mandi. It uses computer vision and
facial behaviour analysis to quantify study focus without any
wearable hardware — just your webcam.

**Key technical contributions:**
- Multi-signal weighted fusion with configurable weights
- Exponential Moving Average smoothing (alpha = 0.08)
- Personal gaze calibration via homography
- Automated PDF session reports with embedded charts
- SQLite session persistence with full event logging
    ''')