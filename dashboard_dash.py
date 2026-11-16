import re
from functools import lru_cache
from pathlib import Path
from typing import Tuple, List, Dict

import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from ai_assistant import (
    generate_diagnostic,
    generate_recommendations,
)
from sentiment_ml import (
    get_student_sentiment,
    get_student_comment_details,
    list_students_with_comments,
    load_sentiment_scores,
    academic_status,
    merge_final_status,
    get_model_label,
    RISK_THRESHOLD_CRITICAL,
)

LABEL_FR = {
    'negative': 'Négatif — risque de décrochage',
    'warning': 'À surveiller',
    'positive': 'Positif',
    'neutral': 'Neutre',
}

DATA_DIR = Path(__file__).resolve().parent / 'data'
STUDENT_FILE = DATA_DIR / 'dataEtd.csv'
EMPLOI_FILE = DATA_DIR / 'emploi.csv'
SEMESTRES = ['S1', 'S2', 'S3', 'S4', 'S5']

SENTIMENT_SCORES = load_sentiment_scores()

COLORS = {
    'background': '#1a1a1a',
    'card': '#262626',
    'text': '#ffffff',
    'muted': '#b3b3b3',
    'success': '#4caf50',
    'warning': '#ffa534',
    'danger': '#ff5c5c',
    'border': '#333333',
}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'borderRadius': '8px',
    'padding': '20px',
    'boxShadow': '0 2px 10px rgba(0,0,0,0.25)',
    'boxSizing': 'border-box',
    'width': '100%',
    'minWidth': 0,
    'minHeight': 0,
    'maxWidth': '100%',
    'overflow': 'hidden',
    'position': 'relative',
}

# Conteneur interne qui borne verticalement la zone scrollable d'une carte.
# On utilise une hauteur explicite (pas de flex % qui ne se résout pas sans parent défini).
CARD_BODY_STYLE = {
    'overflowY': 'auto',
    'overflowX': 'hidden',
    'minHeight': 0,
    'width': '100%',
    'boxSizing': 'border-box',
}

CARD_FIXED_STYLE = {
    **CARD_STYLE,
    'display': 'flex',
    'flexDirection': 'column',
    'height': '360px',
    'maxHeight': '360px',
}

CARD_FIXED_SMALL_STYLE = {
    **CARD_STYLE,
    'display': 'flex',
    'flexDirection': 'column',
    'height': '240px',
    'maxHeight': '240px',
}

CARD_INNER_STYLE = {
    'backgroundColor': '#1f1f1f',
    'borderRadius': '8px',
    'padding': '16px',
    'boxSizing': 'border-box',
}

SECTION_TITLE = {
    'fontSize': 16,
    'fontWeight': 700,
    'color': COLORS['text'],
    'marginBottom': 6,
}

SECTION_SUBTITLE = {
    'color': COLORS['muted'],
    'fontSize': 12,
    'marginBottom': 14,
    'lineHeight': 1.4,
}

DROPDOWN_STYLE = {
    'width': '100%',
    'color': '#111111',
}

PANEL_SCROLL_HEIGHT = '400px'

LAYOUT_GAP = '32px'

ROW_2COL_STYLE = {
    'display': 'grid',
    'gridTemplateColumns': 'minmax(0, 1fr) minmax(0, 1fr)',
    'gap': LAYOUT_GAP,
    'marginBottom': LAYOUT_GAP,
    'width': '100%',
    'boxSizing': 'border-box',
    'minWidth': 0,
    'isolation': 'isolate',
}

ROW_KPI_STYLE = {
    'display': 'grid',
    'gridTemplateColumns': 'repeat(3, minmax(0, 1fr))',
    'gap': LAYOUT_GAP,
    'marginBottom': LAYOUT_GAP,
    'width': '100%',
    'boxSizing': 'border-box',
    'minWidth': 0,
}

COL_CELL_STYLE = {
    'minWidth': 0,
    'width': '100%',
    'maxWidth': '100%',
    'boxSizing': 'border-box',
    'overflow': 'hidden',
    'display': 'flex',
    'flexDirection': 'column',
}

KPI_TITLE_STYLE = {
    'gridColumn': '1 / -1',
    'color': COLORS['muted'],
    'fontSize': 13,
    'fontWeight': 600,
    'marginBottom': 4,
}

PAGE_STYLE = {
    'minHeight': '100vh',
    'padding': '32px',
    'backgroundColor': COLORS['background'],
    'fontFamily': 'Inter, Helvetica, Arial, sans-serif',
    'maxWidth': '1400px',
    'margin': '0 auto',
    'boxSizing': 'border-box',
    'overflowX': 'hidden',
}


def _merge_style(base: Dict, extra: Dict = None) -> Dict:
    merged = dict(base)
    if extra:
        merged.update(extra)
    return merged


def _card(children, extra_style: Dict = None) -> html.Div:
    return html.Div(children, style=_merge_style(CARD_STYLE, extra_style))


def _scroll_body(children, height: str = '280px') -> html.Div:
    """Conteneur scrollable à hauteur fixe explicite."""
    return html.Div(
        children,
        style={
            'height': height,
            'maxHeight': height,
            'overflowY': 'auto',
            'overflowX': 'hidden',
            'minHeight': 0,
            'width': '100%',
            'boxSizing': 'border-box',
        },
    )


def _scroll_body_flex(children, height: str = '180px') -> html.Div:
    """Conteneur scrollable à hauteur fixe explicite (variante)."""
    return _scroll_body(children, height=height)


def _content_body(children) -> html.Div:
    """Fluid body for cards that should grow with their content."""
    return html.Div(
        children,
        style={
            'width': '100%',
            'boxSizing': 'border-box',
            'minHeight': 0,
        },
    )


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_students = pd.read_csv(STUDENT_FILE)
    df_timetable = pd.read_csv(EMPLOI_FILE)
    return df_students, df_timetable


def semester_grade_columns(df: pd.DataFrame, semester: str) -> List[str]:
    pattern = re.compile(rf'^{semester}_.+_Note_Module_Final$')
    return [column for column in df.columns if pattern.match(column)]


def module_label(column: str, semester: str) -> str:
    label = column.replace(f'{semester}_', '').replace('_', ' ')
    return label.replace('Note Module Final', '').strip()


def compute_semester_stats(df: pd.DataFrame, semester: str, timetable: pd.DataFrame) -> Dict:
    columns = semester_grade_columns(df, semester)
    if not columns:
        return {
            'success_rate': 0.0,
            'average': 0.0,
            'absence_rate': 0.0,
            'risk_count': 0,
            'academic_risk_count': 0,
            'semester_fail_count': 0,
            'sentiment_risk_count': 0,
            'total_students': 0,
            'student_rows': pd.DataFrame(),
            'module_mean': pd.Series(dtype=float),
            'schedule_count': 0,
            'schedule_top_day': 'Aucun',
            'schedule': timetable.iloc[0:0],
            'module_validation': {},
            'semester': semester,
        }

    grades = df[columns].apply(pd.to_numeric, errors='coerce')
    semester_average = grades.mean(axis=1)
    has_semester_grades = grades.notna().any(axis=1)
    attendance = pd.to_numeric(df['Absences_heures'], errors='coerce').fillna(0)

    valid_avg = semester_average[has_semester_grades]
    module_mean = grades.loc[has_semester_grades].mean().rename(
        index={col: module_label(col, semester) for col in columns}
    )
    success_rate = float((valid_avg >= 10).mean() * 100) if len(valid_avg) else 0.0
    absence_rate = float((attendance[has_semester_grades] > 18).mean() * 100) if has_semester_grades.any() else 0.0
    average = float(valid_avg.mean()) if len(valid_avg) else 0.0
    semester_fail_count = int((valid_avg < 10).sum())
    total_students = int(has_semester_grades.sum())

    student_rows = df.loc[has_semester_grades, ['Apogee', 'Nom', 'Prenom', 'Absences_heures']].copy()
    student_rows['Apogee'] = student_rows['Apogee'].astype(str)
    student_rows['Moyenne'] = semester_average[has_semester_grades].round(2).values
    student_rows['Score'] = (
        (10 - semester_average[has_semester_grades]).clip(lower=0) + attendance[has_semester_grades] * 0.08
    ).fillna(0).values

    statuts_academiques = []
    statuts_finaux = []
    motifs = []
    risk_scores = []
    extraits = []
    sentiment_flags = []

    for _, row in student_rows.iterrows():
        moy = float(row['Moyenne'])
        abs_h = float(row['Absences_heures'])
        statut_acad = academic_status(moy, abs_h, semester_view=True)
        sentiment = get_student_sentiment(row['Apogee'], semester)
        statut_final, motif = merge_final_status(statut_acad, sentiment, moyenne=moy, absences=abs_h)
        statuts_academiques.append(statut_acad)
        statuts_finaux.append(statut_final)
        motifs.append(motif)
        risk_scores.append(sentiment['risk_score'])
        extraits.append(sentiment.get('extrait_alerte', ''))
        sentiment_flags.append(bool(sentiment.get('sentiment_alert')))

    student_rows['Statut_academique'] = statuts_academiques
    student_rows['Statut'] = statuts_finaux
    student_rows['Motif_alerte'] = motifs
    student_rows['Sentiment_risk'] = risk_scores
    student_rows['Sentiment_extrait'] = extraits
    student_rows['Alerte_sentiment'] = sentiment_flags
    student_rows['Score'] = student_rows['Score'] + student_rows['Sentiment_risk'] * 4

    academic_risk = int((student_rows['Statut_academique'] == 'Critique').sum())
    sentiment_risk_count = int(student_rows['Alerte_sentiment'].sum())
    risk_count = int((student_rows['Statut'] == 'Critique').sum())

    alert_candidates = student_rows[
        (student_rows['Statut'] != 'Ok')
        | (student_rows['Sentiment_risk'] >= RISK_THRESHOLD_CRITICAL * 0.85)
    ]
    student_rows = alert_candidates.sort_values(['Score', 'Moyenne'], ascending=[False, True]).head(12)

    schedule = timetable[timetable['Semaine'] == semester]
    schedule_count = len(schedule)
    top_day = schedule['Jour'].value_counts().idxmax() if schedule_count else 'Aucun'

    # Module validation stats
    module_validation = {}
    grades_active = grades.loc[has_semester_grades]
    for col in columns:
        module_grades = grades_active[col].dropna()
        total_mod = len(module_grades)
        validated = int((module_grades >= 10).sum())
        makeup = int(((module_grades >= 7) & (module_grades < 10)).sum())
        failed = int((module_grades < 7).sum())
        module_label_name = module_label(col, semester)
        module_validation[module_label_name] = {
            'validated': validated,
            'makeup': makeup,
            'failed': failed,
            'total': total_mod,
        }

    return {
        'success_rate': round(success_rate, 1),
        'average': round(average, 2),
        'absence_rate': round(absence_rate, 1),
        'risk_count': risk_count,
        'academic_risk_count': academic_risk,
        'semester_fail_count': semester_fail_count,
        'sentiment_risk_count': sentiment_risk_count,
        'total_students': total_students,
        'student_rows': student_rows,
        'module_mean': module_mean.sort_values(),
        'schedule_count': schedule_count,
        'schedule_top_day': top_day,
        'schedule': schedule,
        'module_validation': module_validation,
        'semester': semester,
    }


def make_card(title: str, value: str, note: str, note_color: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={'color': COLORS['muted'], 'fontSize': 14, 'marginBottom': 12}),
            html.Div(value, style={'color': COLORS['text'], 'fontSize': 42, 'fontWeight': 700, 'marginBottom': 10}),
            html.Div(note, style={'color': note_color, 'fontSize': 14}),
        ],
        style=_merge_style(CARD_STYLE, {
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'space-between',
            'minHeight': '132px',
            'height': '100%',
        }),
    )


def make_progress_row(label: str, percent: float) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label, style={'fontWeight': 600, 'color': COLORS['text']}),
                    html.Span(f" {percent:.1f}%", style={'color': COLORS['muted'], 'float': 'right'}),
                ],
                style={'marginBottom': '8px', 'fontSize': 14},
            ),
            html.Div(
                style={
                    'height': '10px',
                    'backgroundColor': '#333333',
                    'borderRadius': '999px',
                    'overflow': 'hidden',
                },
                children=[
                    html.Div(
                        style={
                            'width': f'{max(min(percent, 100), 0)}%',
                            'height': '100%',
                            'backgroundColor': COLORS['success'] if percent >= 12 else COLORS['warning'] if percent >= 8 else COLORS['danger'],
                            'borderRadius': '999px',
                        }
                    )
                ],
            ),
        ],
        style={'marginBottom': '16px'},
    )


def make_alert_item(row: pd.Series) -> html.Div:
    status_color = COLORS['danger'] if row['Statut'] == 'Critique' else COLORS['warning'] if row['Statut'] == 'Attention' else COLORS['success']
    sentiment_only = bool(row.get('Alerte_sentiment')) and row.get('Statut_academique') not in ('Critique',)
    badge_text = 'Critique · sentiment' if sentiment_only and row['Statut'] == 'Critique' else row['Statut']
    sentiment_line = None
    if row.get('Sentiment_extrait'):
        sentiment_line = html.Div(
            f"🧠 Risque texte ({row.get('Sentiment_risk', 0):.0%}) : {row['Sentiment_extrait']}",
            style={'color': COLORS['warning'] if row.get('Alerte_sentiment') else COLORS['muted'], 'fontSize': 12, 'marginTop': 4, 'lineHeight': 1.35},
        )
    return html.Div(
        [
            html.Div(
                [
                    html.Div(f"{row['Nom']} {row['Prenom']}", style={'fontWeight': 700, 'color': COLORS['text'], 'marginBottom': 4}),
                    html.Div(f"moy. {row['Moyenne']:.2f} · abs. {int(row['Absences_heures'])}h · {row.get('Motif_alerte', '—')}", style={'color': COLORS['muted'], 'fontSize': 13}),
                    sentiment_line,
                ],
                style={'flex': '1', 'paddingRight': 12},
            ),
            html.Div(
                badge_text,
                style={
                    'padding': '6px 12px',
                    'borderRadius': '999px',
                    'backgroundColor': status_color,
                    'color': '#111111',
                    'fontWeight': 700,
                    'fontSize': 12,
                    'whiteSpace': 'nowrap',
                },
            ),
        ],
        style={
            'display': 'flex',
            'alignItems': 'flex-start',
            'justifyContent': 'space-between',
            'padding': '14px 16px',
            'borderBottom': f'1px solid {COLORS["border"]}',
        },
    )


def build_kpi_cards(stats: Dict, semester: str) -> html.Div:
    success_note_color = COLORS['success'] if stats['success_rate'] >= 80 else COLORS['warning'] if stats['success_rate'] >= 60 else COLORS['danger']
    average_note_color = COLORS['success'] if stats['average'] >= 11 else COLORS['warning'] if stats['average'] >= 10 else COLORS['danger']
    risk_note_color = COLORS['danger'] if stats['risk_count'] > max(stats['total_students'], 1) * 0.25 else COLORS['warning']
    sentiment_count = stats.get('sentiment_risk_count', 0)
    sentiment_color = COLORS['danger'] if sentiment_count > 0 else COLORS['success']
    fail_count = stats.get('semester_fail_count', 0)
    cards = [
        make_card('Taux de réussite', f"{stats['success_rate']:.0f}%", f'Moyenne semestrielle ≥ 10 · {semester}', success_note_color),
        make_card('Moyenne générale', f"{stats['average']:.2f}", f'Uniquement modules {semester}', average_note_color),
        make_card('Échecs au semestre', f"{fail_count}", f'Moyenne < 10 · {semester}', COLORS['danger'] if fail_count > 0 else COLORS['success']),
        make_card('Étudiants en risque', f"{stats['risk_count']}", f'Critique · {semester}', risk_note_color),
        make_card('Alertes décrochage', f"{sentiment_count}", f'Commentaires {semester}', sentiment_color),
        make_card('Absences élevées', f"{stats['absence_rate']:.0f}%", f'> 18h annuel · filière', COLORS['warning'] if stats['absence_rate'] > 20 else COLORS['muted']),
    ]
    return html.Div(
        [
            html.Div(
                f'Indicateurs — {semester} ({stats["total_students"]} étudiants avec notes ce semestre)',
                style=KPI_TITLE_STYLE,
            ),
            *cards,
        ],
        style=ROW_KPI_STYLE,
    )


def build_module_performance(stats: Dict) -> html.Div:
    return _card(
        [
            html.Div('RÉUSSITE PAR MODULE', style={**SECTION_TITLE, 'marginBottom': 12, 'flexShrink': 0}),
            _scroll_body(
                [make_progress_row(label, value) for label, value in stats['module_mean'].items()],
                height='300px',
            ),
        ],
        CARD_FIXED_STYLE,
    )


def student_semester_average(df: pd.DataFrame, apogee: str, semester: str) -> float:
    columns = semester_grade_columns(df, semester)
    if not columns:
        return float('nan')
    row = df[df['Apogee'].astype(str) == str(apogee)]
    if row.empty:
        return float('nan')
    grades = pd.to_numeric(row.iloc[0][columns], errors='coerce')
    return float(grades.mean())


def _sentiment_risk_bar(risk: float) -> html.Div:
    color = COLORS['danger'] if risk >= RISK_THRESHOLD_CRITICAL else COLORS['warning'] if risk >= 0.45 else COLORS['success']
    return html.Div(
        style={'height': '8px', 'backgroundColor': '#333', 'borderRadius': '999px', 'overflow': 'hidden', 'marginTop': 6},
        children=[
            html.Div(style={'width': f'{risk * 100:.0f}%', 'height': '100%', 'backgroundColor': color, 'borderRadius': '999px'}),
        ],
    )


def build_sentiment_detail_panel(apogee: str, semester: str, df: pd.DataFrame) -> html.Div:
    row = df[df['Apogee'].astype(str) == str(apogee)]
    if row.empty:
        return html.Div('Étudiant introuvable.', style={'color': COLORS['muted']})

    student = row.iloc[0]
    moyenne = student_semester_average(df, apogee, semester)
    absences = float(student['Absences_heures'])
    statut_acad = academic_status(moyenne, absences, semester_view=True) if pd.notna(moyenne) else '—'
    sentiment = get_student_sentiment(apogee, semester)
    statut_final, motif = merge_final_status(
        statut_acad if statut_acad != '—' else 'Ok',
        sentiment,
        moyenne=moyenne if pd.notna(moyenne) else None,
        absences=absences,
    )
    details = get_student_comment_details(apogee, semester)

    alert_color = COLORS['danger'] if sentiment.get('sentiment_alert') else COLORS['warning'] if sentiment['risk_score'] >= 0.45 else COLORS['success']

    comment_blocks = []
    for item in details:
        label_fr = LABEL_FR.get(item['label'], item['label'])
        comment_blocks.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(item['source'], style={'fontWeight': 700, 'color': COLORS['text']}),
                            html.Span(f" · {item['date']}", style={'color': COLORS['muted'], 'marginLeft': 8, 'fontSize': 12}),
                            html.Span(
                                f" · {label_fr} ({item['risk_score']:.0%})",
                                style={'color': alert_color if item['risk_score'] >= 0.45 else COLORS['muted'], 'marginLeft': 8, 'fontSize': 12},
                            ),
                        ],
                        style={'marginBottom': 8},
                    ),
                    html.Div(item['commentaire'], style={'color': COLORS['muted'], 'fontSize': 13, 'lineHeight': 1.45, 'marginBottom': 4}),
                    _sentiment_risk_bar(item['risk_score']),
                    html.Div(f"Méthode : {item['method']}", style={'color': '#666', 'fontSize': 11, 'marginTop': 6}),
                ],
                style={
                    'padding': '14px',
                    'backgroundColor': '#1f1f1f',
                    'borderRadius': 8,
                    'marginBottom': 12,
                    'borderLeft': f'3px solid {COLORS["danger"] if item["risk_score"] >= RISK_THRESHOLD_CRITICAL else COLORS["warning"] if item["risk_score"] >= 0.45 else COLORS["success"]}',
                },
            )
        )

    if not comment_blocks:
        comment_blocks = [
            html.Div('Aucun commentaire enregistré pour cet étudiant sur ce semestre.', style={'color': COLORS['muted'], 'fontSize': 13}),
        ]

    return html.Div(
        [
            html.Div('DÉTAIL SENTIMENT — EARLY WARNING', style={**SECTION_TITLE, 'marginBottom': 8}),
            html.Div(get_model_label(), style={'color': COLORS['muted'], 'fontSize': 12, 'marginBottom': 16}),
            html.Div(
                [
                    html.Div(f"{student['Nom']} {student['Prenom']}", style={'fontSize': 20, 'fontWeight': 700, 'color': COLORS['text']}),
                    html.Div(f"Apogée {apogee} · {semester}", style={'color': COLORS['muted'], 'fontSize': 13, 'marginTop': 4}),
                ],
                style={'marginBottom': 16},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div('Risque global (texte)', style={'color': COLORS['muted'], 'fontSize': 12}),
                            html.Div(f"{sentiment['risk_score']:.0%}", style={'fontSize': 28, 'fontWeight': 700, 'color': alert_color}),
                            _sentiment_risk_bar(sentiment['risk_score']),
                        ],
                        style={'flex': 1, 'minWidth': '140px'},
                    ),
                    html.Div(
                        [
                            html.Div('Statut académique', style={'color': COLORS['muted'], 'fontSize': 12}),
                            html.Div(statut_acad if pd.notna(moyenne) else '—', style={'fontSize': 18, 'fontWeight': 700, 'color': COLORS['text']}),
                            html.Div(f"Moy. {moyenne:.2f}" if pd.notna(moyenne) else '', style={'color': COLORS['muted'], 'fontSize': 12}),
                        ],
                        style={'flex': 1, 'minWidth': '120px'},
                    ),
                    html.Div(
                        [
                            html.Div('Statut final', style={'color': COLORS['muted'], 'fontSize': 12}),
                            html.Div(statut_final, style={'fontSize': 18, 'fontWeight': 700, 'color': alert_color}),
                            html.Div(motif, style={'color': COLORS['muted'], 'fontSize': 12}),
                        ],
                        style={'flex': 1, 'minWidth': '140px'},
                    ),
                ],
                style=_merge_style(CARD_INNER_STYLE, {
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(3, minmax(0, 1fr))',
                    'gap': '16px',
                    'marginBottom': 20,
                }),
            ),
            html.Div('Commentaires analysés', style={'fontSize': 14, 'fontWeight': 600, 'color': COLORS['text'], 'marginBottom': 12}),
            html.Div(comment_blocks),
        ],
    )


def build_alert_panel(stats: Dict) -> html.Div:
    return _card(
        [
            html.Div('ÉTUDIANTS EN ALERTE', style={**SECTION_TITLE, 'flexShrink': 0}),
            html.Div(
                'Statut fusionné : notes, absences et analyse de sentiment des commentaires.',
                style={**SECTION_SUBTITLE, 'flexShrink': 0},
            ),
            _scroll_body(
                [make_alert_item(row) for _, row in stats['student_rows'].iterrows()],
                height='190px',
            ),
            html.Div(
                [
                    html.Div('Emploi du temps', style={'fontSize': 13, 'color': COLORS['muted'], 'marginBottom': 6}),
                    html.Div(f"Séances détectées : {stats['schedule_count']}", style={'color': COLORS['text'], 'fontSize': 12, 'marginBottom': 3}),
                    html.Div(f"Jour le plus chargé : {stats['schedule_top_day']}", style={'color': COLORS['text'], 'fontSize': 12, 'marginBottom': 3}),
                    html.Div(f"Absences > 18h (annuel) : {stats['absence_rate']:.0f}%", style={'color': COLORS['text'], 'fontSize': 12}),
                ],
                style=_merge_style(CARD_INNER_STYLE, {'marginTop': '10px', 'flexShrink': 0, 'padding': '10px 12px'}),
            ),
        ],
        CARD_FIXED_STYLE,
    )


def build_ai_diagnostic_card(stats: Dict, semester: str) -> html.Div:
    """
    Construit la carte de diagnostic IA - Niveau DÉTECTER
    Analyse les KPIs et identifie les problèmes
    """
    diagnostic = generate_diagnostic(stats, semester, use_ollama=True)
    
    alert_color = {
        'success': '#4caf50',
        'warning': '#ffa534',
        'danger': '#ff5c5c',
    }.get(diagnostic['alert_level'], COLORS['warning'])
    
    status_icon = {
        'success': '✓',
        'warning': '⚠',
        'danger': '🔴',
    }.get(diagnostic['alert_level'], '?')

    return html.Div(
        [
            html.Div(
                [
                    html.Span(status_icon, style={'fontSize': 20, 'marginRight': 8}),
                    html.Span('Diagnostic IA', style={'fontSize': 14, 'fontWeight': 700, 'color': COLORS['text']}),
                ],
                style={'display': 'flex', 'alignItems': 'center', 'marginBottom': 8, 'flexShrink': 0},
            ),
            html.Div(
                diagnostic['diagnostic'],
                style={
                    'color': COLORS['text'],
                    'fontSize': 13,
                    'lineHeight': 1.5,
                    'overflowY': 'auto',
                    'minHeight': 0,
                    'flex': '1 1 auto',
                    'maxHeight': '100%',
                    'paddingRight': '4px',
                },
            ),
        ],
        style=_merge_style(CARD_STYLE, {
            'borderLeft': f'4px solid {alert_color}',
            'display': 'flex',
            'flexDirection': 'column',
            'minHeight': '200px',
            'height': 'auto',
        }),
    )


def build_ai_recommendations_card(stats: Dict, semester: str, alert_students: pd.DataFrame) -> html.Div:
    """
    Construit la carte de recommandations IA - Niveaux CIBLER, COMPRENDRE, AGIR
    Fournit des actions concrètes et actionnables
    """
    recommendations = generate_recommendations(stats, semester, alert_students, use_ollama=True)
    
    recommendation_items = []
    for i, rec in enumerate(recommendations[:3]):  # Limiter à 3 recommandations
        rec_item = html.Div(
            [
                html.Div(
                    [
                        html.Div(rec['title'], style={'fontWeight': 700, 'color': COLORS['text'], 'marginBottom': 6}),
                        html.Div(rec['description'], style={'color': COLORS['muted'], 'fontSize': 13, 'marginBottom': 8, 'lineHeight': 1.4}),
                        html.Div(
                            [
                                html.Div(
                                    f"→ {action}",
                                    style={
                                        'color': rec['status_color'],
                                        'fontSize': 12,
                                        'fontWeight': 600,
                                        'marginTop': 6,
                                        'padding': '8px 12px',
                                        'backgroundColor': '#1f1f1f',
                                        'borderRadius': 4,
                                        'borderLeft': f'3px solid {rec["status_color"]}',
                                    },
                                )
                                for action in rec['actions']
                            ],
                            style={'marginTop': 10},
                        ),
                    ],
                    style={
                        'padding': '12px 14px',
                        'backgroundColor': '#1f1f1f',
                        'borderRadius': 6,
                        'borderLeft': f'3px solid {rec["status_color"]}',
                        'marginBottom': 10 if i < len(recommendations) - 1 else 0,
                    },
                )
            ]
        )
        recommendation_items.append(rec_item)

    return html.Div(
        [
            html.Div(
                [
                    html.Span('💡', style={'fontSize': 18, 'marginRight': 8}),
                    html.Span('Recommandations IA', style={'fontSize': 14, 'fontWeight': 700}),
                ],
                style={'display': 'flex', 'alignItems': 'center', 'marginBottom': 16, 'color': COLORS['text'], 'flexShrink': 0},
            ),
            _content_body(recommendation_items),
        ],
        style=_merge_style(CARD_STYLE, {
            'flex': '1',
            'display': 'flex',
            'flexDirection': 'column',
            'minHeight': '280px',
            'height': 'auto',
        }),
    )


def build_pie_chart(stats: Dict, selected_module: str = None) -> dcc.Graph:
    """Pie chart showing student validation status for a specific module."""
    if selected_module is None or selected_module not in stats['module_validation']:
        selected_module = list(stats['module_validation'].keys())[0] if stats['module_validation'] else None
    
    if selected_module is None:
        return dcc.Graph(figure={'data': [], 'layout': {'title': 'Aucun module disponible'}}, style={'height': '380px'})

    mod_data = stats['module_validation'][selected_module]
    validation_counts = {
        'validé': mod_data['validated'],
        'ratt': mod_data['makeup'],
        'non-validé': mod_data['failed'],
    }

    fig = {
        'data': [{
            'labels': ['✓ Validés', '⚠ Rattrapage', '✗ Non validés'],
            'values': [validation_counts['validé'], validation_counts['ratt'], validation_counts['non-validé']],
            'type': 'pie',
            'marker': {'colors': [COLORS['success'], COLORS['warning'], COLORS['danger']]},
            'textinfo': 'label+percent+value',
            'hoverinfo': 'label+value+percent',
        }],
        'layout': {
            'title': f'Validation : {selected_module}',
            'font': {'color': COLORS['text'], 'size': 12},
            'paper_bgcolor': COLORS['card'],
            'plot_bgcolor': COLORS['card'],
            'margin': {'l': 0, 'r': 0, 't': 40, 'b': 0},
        },
    }
    return dcc.Graph(
        figure=fig,
        style={'height': '320px', 'width': '100%', 'maxWidth': '100%'},
        config={'responsive': True, 'displayModeBar': False},
    )


def build_module_detail_card(stats: Dict, selected_module: str = None) -> html.Div:
    if selected_module is None or selected_module not in stats['module_validation']:
        return html.Div(
            'Cliquez sur un module de la pie chart pour voir les détails.',
            style={'color': COLORS['muted'], 'padding': '20px', 'textAlign': 'center', 'fontSize': 14},
        )

    mod_data = stats['module_validation'][selected_module]
    total = mod_data['total']
    validated_pct = (mod_data['validated'] / total * 100) if total > 0 else 0
    makeup_pct = (mod_data['makeup'] / total * 100) if total > 0 else 0
    failed_pct = (mod_data['failed'] / total * 100) if total > 0 else 0

    return html.Div(
        [
            html.Div(f'Détails : {selected_module}', style={'fontSize': 16, 'fontWeight': 700, 'color': COLORS['text'], 'marginBottom': 16}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div('✓ Validés', style={'color': COLORS['success'], 'fontWeight': 700, 'marginBottom': 4}),
                            html.Div(f"{mod_data['validated']} élèves ({validated_pct:.1f}%)", style={'color': COLORS['text']}),
                        ],
                        style={'padding': '12px', 'backgroundColor': '#1f1f1f', 'borderRadius': 6, 'marginBottom': 10},
                    ),
                    html.Div(
                        [
                            html.Div('⚠ Rattrapage', style={'color': COLORS['warning'], 'fontWeight': 700, 'marginBottom': 4}),
                            html.Div(f"{mod_data['makeup']} élèves ({makeup_pct:.1f}%)", style={'color': COLORS['text']}),
                        ],
                        style={'padding': '12px', 'backgroundColor': '#1f1f1f', 'borderRadius': 6, 'marginBottom': 10},
                    ),
                    html.Div(
                        [
                            html.Div('✗ Non validés', style={'color': COLORS['danger'], 'fontWeight': 700, 'marginBottom': 4}),
                            html.Div(f"{mod_data['failed']} élèves ({failed_pct:.1f}%)", style={'color': COLORS['text']}),
                        ],
                        style={'padding': '12px', 'backgroundColor': '#1f1f1f', 'borderRadius': 6},
                    ),
                ],
            ),
        ],
        style=CARD_STYLE,
    )


def build_pie_section(stats: Dict, default_module: str = None) -> html.Div:
    modules = list(stats['module_validation'].keys())
    selected = default_module if default_module in modules else (modules[0] if modules else None)
    return _card(
        [
            html.Div('Sélectionner un module', style={'fontSize': 13, 'color': COLORS['muted'], 'marginBottom': 6, 'flexShrink': 0}),
            dcc.Dropdown(
                id='module-dropdown',
                options=[{'label': m, 'value': m} for m in modules],
                value=selected,
                clearable=False,
                style=_merge_style(DROPDOWN_STYLE, {'marginBottom': 10, 'fontSize': 13}),
            ),
            html.Div(id='pie-chart-content', children=[build_pie_chart(stats, selected)], style={'flex': '0 0 auto'}),
        ],
        CARD_FIXED_STYLE,
    )


def build_sentiment_section() -> html.Div:
    return _card(
        [
            html.Div('Analyser un étudiant', style={'fontSize': 13, 'color': COLORS['muted'], 'marginBottom': 6, 'flexShrink': 0}),
            dcc.Dropdown(
                id='sentiment-student-dropdown',
                clearable=False,
                style=_merge_style(DROPDOWN_STYLE, {'marginBottom': 10, 'fontSize': 13}),
            ),
            html.Div(
                id='sentiment-detail-panel',
                style={'flex': '1 1 auto', 'overflowY': 'auto', 'minHeight': 0, 'maxHeight': '100%', 'width': '100%'},
            ),
        ],
        CARD_FIXED_STYLE,
    )


df_students, df_timetable = load_data()


@lru_cache(maxsize=8)
def get_cached_semester_stats(semester: str) -> Dict:
    return compute_semester_stats(df_students, semester, df_timetable)


app = Dash(__name__)
app.title = 'Dashboard INDIA'

app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            *, *::before, *::after { box-sizing: border-box; }
            .js-plotly-plot, .plot-container.plotly { max-width: 100% !important; }
            body { overflow-x: hidden; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

stats = get_cached_semester_stats('S1')

_default_module = list(stats['module_validation'].keys())[0] if stats['module_validation'] else None

app.layout = html.Div(
    [
        html.Div(
            html.Div('Dashboard filière INDIA', style={'color': COLORS['text'], 'fontSize': 32, 'fontWeight': 700}),
            style={'marginBottom': '20px'},
        ),
        html.Div(
            dcc.Dropdown(
                id='semester-dropdown',
                options=[{'label': semester, 'value': semester} for semester in SEMESTRES],
                value='S1',
                clearable=False,
                style={'width': '240px', 'color': '#111111'},
            ),
            style={'marginBottom': '26px'},
        ),
        html.Div(id='kpi-row', children=build_kpi_cards(stats, 'S1')),
        html.Div(
            [
                html.Div(
                    id='ai-diagnostic-box',
                    style=COL_CELL_STYLE,
                ),
                html.Div(
                    id='ai-recommendations-box',
                    style=COL_CELL_STYLE,
                ),
            ],
            style=ROW_2COL_STYLE,
        ),
        html.Div(
            [
                html.Div(
                    id='left-panel',
                    style=COL_CELL_STYLE,
                    children=[build_module_performance(stats)],
                ),
                html.Div(
                    id='right-panel',
                    style=COL_CELL_STYLE,
                    children=[build_alert_panel(stats)],
                ),
            ],
            style=ROW_2COL_STYLE,
        ),
        html.Div(
            [
                html.Div(
                    id='pie-container',
                    style=COL_CELL_STYLE,
                    children=[build_pie_section(stats, _default_module)],
                ),
                html.Div(
                    id='sentiment-column',
                    style=COL_CELL_STYLE,
                    children=[build_sentiment_section()],
                ),
            ],
            style=ROW_2COL_STYLE,
        ),
    ],
    style=PAGE_STYLE,
)


@app.callback(
    Output('module-dropdown', 'options'),
    Output('module-dropdown', 'value'),
    Input('semester-dropdown', 'value'),
)
def update_module_dropdown(selected_semester: str):
    stats = get_cached_semester_stats(selected_semester)
    modules = list(stats['module_validation'].keys())
    options = [{'label': m, 'value': m} for m in modules]
    value = modules[0] if modules else None
    return options, value


@app.callback(
    Output('kpi-row', 'children'),
    Output('left-panel', 'children'),
    Output('right-panel', 'children'),
    Input('semester-dropdown', 'value'),
    prevent_initial_call=True,
)
def update_semester_dashboard(selected_semester: str):
    stats = get_cached_semester_stats(selected_semester)
    return (
        build_kpi_cards(stats, selected_semester),
        [build_module_performance(stats)],
        [build_alert_panel(stats)],
    )


@app.callback(
    Output('pie-chart-content', 'children'),
    Input('semester-dropdown', 'value'),
    Input('module-dropdown', 'value'),
)
def update_pie_chart(selected_semester: str, selected_module: str):
    stats = get_cached_semester_stats(selected_semester)
    return [build_pie_chart(stats, selected_module)]


@app.callback(
    Output('ai-diagnostic-box', 'children'),
    Output('ai-recommendations-box', 'children'),
    Input('semester-dropdown', 'value'),
)
def update_ai_cards(selected_semester: str):
    stats = get_cached_semester_stats(selected_semester)
    return (
        [build_ai_diagnostic_card(stats, selected_semester)],
        [build_ai_recommendations_card(stats, selected_semester, stats['student_rows'])],
    )


@app.callback(
    Output('sentiment-student-dropdown', 'options'),
    Output('sentiment-student-dropdown', 'value'),
    Input('semester-dropdown', 'value'),
)
def update_sentiment_student_dropdown(selected_semester: str):
    apogees = list_students_with_comments(selected_semester)
    if not apogees:
        return [], None

    scores = load_sentiment_scores()
    risk_map: Dict[str, float] = {}
    if not scores.empty:
        for _, srow in scores[scores['Semestre'] == selected_semester].iterrows():
            risk_map[str(srow['Apogee'])] = float(srow['risk_score'])

    options = []
    for apogee in apogees:
        row = df_students[df_students['Apogee'].astype(str) == apogee]
        if row.empty:
            label = apogee
        else:
            student = row.iloc[0]
            risk = risk_map.get(apogee, 0.0)
            label = f"{student['Nom']} {student['Prenom']} — risque texte {risk:.0%}"
        options.append({'label': label, 'value': apogee})

    options.sort(key=lambda opt: risk_map.get(opt['value'], 0), reverse=True)
    default = options[0]['value'] if options else None
    return options, default


@app.callback(
    Output('sentiment-detail-panel', 'children'),
    Input('semester-dropdown', 'value'),
    Input('sentiment-student-dropdown', 'value'),
)
def update_sentiment_detail_panel(selected_semester: str, selected_apogee: str):
    if not selected_apogee:
        return html.Div(
            'Aucun commentaire de suivi pour ce semestre. Ajoutez des lignes dans data/commentaires_suivi.csv.',
            style={'color': COLORS['muted'], 'fontSize': 13},
        )
    return build_sentiment_detail_panel(selected_apogee, selected_semester, df_students)


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8052)
