from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable, Image as RLImage
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
from datetime import datetime

NAVY  = colors.HexColor('#1E3A5F')
TEAL  = colors.HexColor('#2E86AB')


def _make_chart(timeline, avg):
    if not timeline:
        return None
    _, scores = zip(*timeline)
    x = np.linspace(0, len(scores) / 30 / 60, len(scores))

    fig, ax = plt.subplots(figsize=(13, 2.8), dpi=150)
    ax.fill_between(x, scores, alpha=0.12, color='#6366f1')
    ax.plot(x, scores, color='#6366f1', linewidth=1.5)
    ax.axhline(
        70, color='#22c55e', linestyle='--',
        linewidth=1, label='Target (70%)'
    )
    ax.axhline(
        avg, color='#f59e0b', linestyle=':',
        linewidth=1.2, label=f'Session avg ({avg:.1f}%)'
    )
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (minutes)', fontsize=9)
    ax.set_ylabel('Focus %', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_report(summary, output_path):
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm
    )
    body = []

    # Title
    body.append(Paragraph(
        'NeuroPose — Study Session Report',
        ParagraphStyle(
            'T', fontSize=22, textColor=NAVY,
            fontName='Helvetica-Bold', spaceAfter=4
        )
    ))
    body.append(
        HRFlowable(width='100%', thickness=2, color=TEAL, spaceAfter=10)
    )

    # Session info table
    start_dt = datetime.fromisoformat(summary['start_time'])
    dur      = summary['duration_s']
    info_data = [
        [
            'Subject',    summary['subject'],
            'Date',       start_dt.strftime('%d %B %Y')
        ],
        [
            'Start time', start_dt.strftime('%H:%M:%S'),
            'Duration',
            f'{dur // 3600:02d}:{(dur % 3600) // 60:02d}:{dur % 60:02d}'
        ],
    ]
    info_table = Table(
        info_data,
        colWidths=[3.5*cm, 6*cm, 3*cm, 5*cm]
    )
    info_table.setStyle(TableStyle([
        ('FONTNAME',      (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME',      (0, 0), (0, -1),  'Helvetica-Bold'),
        ('FONTNAME',      (2, 0), (2, -1),  'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 10),
        ('TEXTCOLOR',     (0, 0), (0, -1),  NAVY),
        ('TEXTCOLOR',     (2, 0), (2, -1),  NAVY),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    body.append(info_table)
    body.append(Spacer(1, 12))

    # KPI row
    kpi_vals = [
        [
            f"{summary['avg_focus']}%",
            f"{summary['peak_focus']}%",
            f"{summary['distractions']}",
            f"{summary['yawns']}",
        ],
        ['Avg Focus', 'Peak Focus', 'Distractions', 'Yawns']
    ]
    kpi_table = Table(kpi_vals, colWidths=[4.25*cm] * 4)
    kpi_table.setStyle(TableStyle([
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',      (0, 0), (-1,  0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1,  0), 20),
        ('FONTSIZE',      (0, 1), (-1,  1), 9),
        ('TEXTCOLOR',     (0, 0), (-1,  0), NAVY),
        ('TEXTCOLOR',     (0, 1), (-1,  1), colors.grey),
        ('BACKGROUND',    (0, 0), (-1, -1), colors.HexColor('#EBF4FA')),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BOX',           (0, 0), (-1, -1), 1, TEAL),
        ('INNERGRID',     (0, 0), (-1, -1), 0.5, colors.lightgrey),
    ]))
    body.append(kpi_table)
    body.append(Spacer(1, 14))

    # Timeline chart
    body.append(Paragraph(
        'Focus Timeline',
        ParagraphStyle(
            'H2', fontSize=13, textColor=NAVY,
            fontName='Helvetica-Bold', spaceAfter=6
        )
    ))
    chart_buf = _make_chart(
        summary.get('timeline', []),
        summary['avg_focus']
    )
    if chart_buf:
        body.append(RLImage(chart_buf, width=16*cm, height=4*cm))
    body.append(Spacer(1, 12))

    # Recommendations
    body.append(Paragraph(
        'Recommendations',
        ParagraphStyle(
            'H2', fontSize=13, textColor=NAVY,
            fontName='Helvetica-Bold', spaceAfter=6
        )
    ))
    recs = []
    if summary['avg_focus'] >= 80:
        recs.append('Excellent session — maintain this consistency!')
    elif summary['avg_focus'] >= 60:
        recs.append(
            'Good focus. Try to reduce distractions '
            'during low-score periods.'
        )
    else:
        recs.append(
            'Focus below target. Try shorter 15-min Pomodoro intervals.'
        )
    if summary['yawns'] > 3:
        recs.append(
            'Multiple fatigue events detected — '
            'ensure 7 to 8 hours of sleep before studying.'
        )
    if summary['distractions'] > 5:
        recs.append(
            'Frequent distractions detected — '
            'remove your phone from your study area.'
        )

    for rec in recs:
        body.append(Paragraph(
            f'• {rec}',
            ParagraphStyle(
                'rec', fontSize=10, spaceAfter=4, leftIndent=10
            )
        ))

    body.append(Spacer(1, 10))
    body.append(
        HRFlowable(width='100%', thickness=1, color=TEAL)
    )
    body.append(Paragraph(
        f'Generated by NeuroPose on '
        f'{datetime.now().strftime("%d %B %Y %H:%M")}',
        ParagraphStyle(
            'ft', fontSize=8,
            textColor=colors.grey, alignment=1
        )
    ))

    doc.build(body)