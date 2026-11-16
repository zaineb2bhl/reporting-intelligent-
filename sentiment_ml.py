"""
Analyse de sentiment (Early Warning) sur les commentaires tuteur / enseignant.
Utilise CamemBERT si disponible, sinon heuristiques lexicales françaises.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data'
COMMENTS_FILE = DATA_DIR / 'commentaires_suivi.csv'
SCORES_FILE = DATA_DIR / 'sentiment_scores.csv'

RISK_THRESHOLD_CRITICAL = 0.65
RISK_THRESHOLD_ATTENTION = 0.45
MODEL_NAME = 'nlptown/bert-base-multilingual-uncased-sentiment'

NEGATIVE_KEYWORDS = (
    'passif', 'passive', 'dépassé', 'dépassée', 'décourag', 'stress', 'anxiété',
    'anxieux', 'démotiv', 'désengag', 'décrochage', 'abandon', 'isolé', 'isolée',
    'frustration', 'difficulté', 'inquiet', 'inquiétude', 'manque de participation',
    'évite', 'silencieu', 'perdu', 'dépassé par le rythme',
)
POSITIVE_KEYWORDS = (
    'motivé', 'motivée', 'engagé', 'engagée', 'excellent', 'progrès', 'positif',
    'dynamique', 'régulier', 'satisfait', 'bonne attitude',
)

_pipeline = None
_pipeline_failed = False


def load_comments() -> pd.DataFrame:
    if not COMMENTS_FILE.exists():
        return pd.DataFrame(columns=['Apogee', 'Semestre', 'Source', 'Commentaire', 'Date_saisie'])
    df = pd.read_csv(COMMENTS_FILE)
    df['Apogee'] = df['Apogee'].astype(str)
    df['Semestre'] = df['Semestre'].astype(str).str.strip()
    return df


def _get_sentiment_pipeline():
    global _pipeline, _pipeline_failed
    if _pipeline_failed:
        return None
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline

        _pipeline = pipeline(
            'sentiment-analysis',
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            truncation=True,
            max_length=256,
        )
    except Exception:
        _pipeline_failed = True
        _pipeline = None
    return _pipeline


def _keyword_risk(text: str) -> float:
    lowered = text.lower()
    risk = 0.18
    negative_hits = sum(1 for word in NEGATIVE_KEYWORDS if word in lowered)
    risk += negative_hits * 0.14
    for word in POSITIVE_KEYWORDS:
        if word in lowered:
            risk -= 0.08
    if negative_hits >= 2:
        risk += 0.12
    return float(min(1.0, max(0.0, risk)))


def _model_risk(text: str) -> Optional[float]:
    pipe = _get_sentiment_pipeline()
    if pipe is None:
        return None
    try:
        result = pipe(text[:512])[0]
        label = result['label'].lower()
        score = float(result['score'])
        if 'star' in label:
            stars = int(label.split()[0])
            if stars <= 2:
                return min(1.0, 0.52 + (2 - stars) * 0.18 + score * 0.3)
            if stars == 3:
                return 0.32 + (1 - score) * 0.12
            return max(0.05, 0.22 - (stars - 3) * 0.06)
        if 'neg' in label or label in {'negative', 'label_0'}:
            return min(1.0, 0.55 + score * 0.45)
        if 'neu' in label or label in {'neutral', 'label_1'}:
            return 0.35 + (1 - score) * 0.15
        return max(0.05, 0.25 - score * 0.15)
    except Exception:
        return None


def analyze_text(text: str, use_transformer: bool = False) -> Dict:
    text = (text or '').strip()
    if not text:
        return {'label': 'neutral', 'confidence': 0.0, 'risk_score': 0.0, 'method': 'empty'}

    keyword = _keyword_risk(text)
    model = _model_risk(text) if use_transformer else None
    if model is not None:
        risk_score = max(keyword, model)
        method = 'transformer+keywords' if keyword > model else 'transformer'
    else:
        risk_score = keyword
        method = 'keywords'

    if risk_score >= RISK_THRESHOLD_CRITICAL:
        label = 'negative'
    elif risk_score >= RISK_THRESHOLD_ATTENTION:
        label = 'warning'
    else:
        label = 'positive' if risk_score < 0.25 else 'neutral'

    return {
        'label': label,
        'confidence': round(risk_score, 3),
        'risk_score': round(risk_score, 3),
        'method': method,
    }


def aggregate_student_semester(
    comments: pd.DataFrame,
    apogee: str,
    semester: str,
    use_transformer: bool = False,
) -> Dict:
    subset = comments[(comments['Apogee'] == str(apogee)) & (comments['Semestre'] == semester)]
    if subset.empty:
        return {
            'risk_score': 0.0,
            'label': 'neutral',
            'nb_commentaires': 0,
            'extrait_alerte': '',
            'sentiment_alert': False,
        }

    analyses = [analyze_text(str(row['Commentaire']), use_transformer=use_transformer) for _, row in subset.iterrows()]
    worst = max(analyses, key=lambda item: item['risk_score'])
    worst_idx = analyses.index(worst)
    excerpt = str(subset.iloc[worst_idx]['Commentaire'])
    if len(excerpt) > 90:
        excerpt = excerpt[:87] + '...'

    return {
        'risk_score': worst['risk_score'],
        'label': worst['label'],
        'nb_commentaires': len(subset),
        'extrait_alerte': excerpt,
        'sentiment_alert': worst['risk_score'] >= RISK_THRESHOLD_CRITICAL,
    }


def compute_all_scores(comments: Optional[pd.DataFrame] = None, use_transformer: bool = True) -> pd.DataFrame:
    comments = comments if comments is not None else load_comments()
    if comments.empty:
        return pd.DataFrame(
            columns=[
                'Apogee', 'Semestre', 'risk_score', 'label',
                'nb_commentaires', 'extrait_alerte', 'sentiment_alert',
            ]
        )

    rows: List[Dict] = []
    for (apogee, semester), _group in comments.groupby(['Apogee', 'Semestre']):
        agg = aggregate_student_semester(comments, str(apogee), str(semester), use_transformer=use_transformer)
        rows.append({'Apogee': str(apogee), 'Semestre': str(semester), **agg})

    return pd.DataFrame(rows)


def save_scores(df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_FILE, index=False)
    return SCORES_FILE


def load_sentiment_scores() -> pd.DataFrame:
    if not SCORES_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SCORES_FILE)
    if df.empty:
        return df
    df['Apogee'] = df['Apogee'].astype(str)
    df['Semestre'] = df['Semestre'].astype(str)
    if 'sentiment_alert' in df.columns:
        df['sentiment_alert'] = df['sentiment_alert'].astype(bool)
    return df


def ensure_sentiment_scores(force: bool = False) -> pd.DataFrame:
    comments = load_comments()
    if comments.empty:
        return pd.DataFrame()

    needs_recompute = force
    if not needs_recompute:
        if not SCORES_FILE.exists():
            needs_recompute = True
        elif COMMENTS_FILE.stat().st_mtime > SCORES_FILE.stat().st_mtime:
            needs_recompute = True

    if needs_recompute:
        scores = compute_all_scores(comments)
        save_scores(scores)
        clear_sentiment_cache()
        return scores
    return load_sentiment_scores()


@lru_cache(maxsize=256)
def get_student_sentiment(apogee: str, semester: str) -> Dict:
    scores = load_sentiment_scores()
    if scores.empty:
        return aggregate_student_semester(load_comments(), str(apogee), semester, use_transformer=False)

    match = scores[(scores['Apogee'] == str(apogee)) & (scores['Semestre'] == semester)]
    if match.empty:
        return aggregate_student_semester(load_comments(), str(apogee), semester, use_transformer=False)

    row = match.iloc[0]
    return {
        'risk_score': float(row['risk_score']),
        'label': str(row['label']),
        'nb_commentaires': int(row['nb_commentaires']),
        'extrait_alerte': str(row.get('extrait_alerte', '') or ''),
        'sentiment_alert': bool(row.get('sentiment_alert', row['risk_score'] >= RISK_THRESHOLD_CRITICAL)),
    }


def academic_status(moyenne: float, absences: float, semester_view: bool = True) -> str:
    """Statut académique. En vue semestre, les absences annuelles ne forcent pas Critique seules."""
    import math
    if semester_view and (moyenne is None or (isinstance(moyenne, float) and math.isnan(moyenne))):
        return '—'
    if moyenne < 10:
        return 'Critique'
    if semester_view:
        if moyenne < 11:
            return 'Attention'
        if absences > 24:
            return 'Attention'
        if absences > 18:
            return 'Attention'
        return 'Ok'
    if absences > 24:
        return 'Critique'
    if moyenne < 11:
        return 'Attention'
    return 'Ok'


def academic_motif_extras(moyenne: float, absences: float) -> str:
    import math
    if moyenne is None or (isinstance(moyenne, float) and math.isnan(moyenne)):
        return ''
    parts = []
    if absences > 24:
        parts.append(f'Absences annuelles {int(absences)}h')
    elif absences > 18:
        parts.append(f'Absences {int(absences)}h')
    return ' · '.join(parts)


def is_transformer_available() -> bool:
    return _get_sentiment_pipeline() is not None


def get_model_label(load_model: bool = False) -> str:
    if SCORES_FILE.exists() and not load_model:
        return 'Scores pré-calculés (fichier sentiment_scores.csv)'
    if load_model and is_transformer_available():
        return f'Transformer ({MODEL_NAME})'
    return 'Heuristiques lexicales (fallback)'


def clear_sentiment_cache() -> None:
    get_student_sentiment.cache_clear()


def get_student_comment_details(apogee: str, semester: str, use_transformer: bool = False) -> List[Dict]:
    comments = load_comments()
    subset = comments[(comments['Apogee'] == str(apogee)) & (comments['Semestre'] == semester)]
    details: List[Dict] = []
    for _, row in subset.iterrows():
        text = str(row['Commentaire'])
        analysis = analyze_text(text, use_transformer=use_transformer)
        details.append({
            'source': str(row.get('Source', '—')),
            'date': str(row.get('Date_saisie', '—')),
            'commentaire': text,
            'label': analysis['label'],
            'risk_score': analysis['risk_score'],
            'method': analysis['method'],
        })
    details.sort(key=lambda item: item['risk_score'], reverse=True)
    return details


def list_students_with_comments(semester: str) -> List[str]:
    comments = load_comments()
    if comments.empty:
        return []
    subset = comments[comments['Semestre'] == semester]
    return sorted(subset['Apogee'].astype(str).unique().tolist())


def merge_final_status(
    statut_academique: str,
    sentiment: Dict,
    moyenne: float | None = None,
    absences: float | None = None,
) -> tuple[str, str]:
    motifs: List[str] = []
    if statut_academique == 'Critique':
        motifs.append('Notes (semestre)')
    elif statut_academique == 'Attention':
        motifs.append('Notes (semestre)')
    extra = academic_motif_extras(moyenne, absences or 0) if moyenne is not None else ''
    if extra:
        for part in extra.split(' · '):
            if part and part not in motifs:
                motifs.append(part)

    risk = float(sentiment.get('risk_score', 0))
    if sentiment.get('sentiment_alert') or risk >= RISK_THRESHOLD_CRITICAL:
        statut_final = 'Critique'
        if statut_academique not in ('Critique', 'Attention'):
            motifs = ['Sentiment (décrochage)'] + [m for m in motifs if m.startswith('Absences')]
        elif 'Sentiment (décrochage)' not in motifs:
            motifs.append('Sentiment (décrochage)')
    elif risk >= RISK_THRESHOLD_ATTENTION and statut_academique == 'Ok':
        statut_final = 'Attention'
        motifs.append('Sentiment')
    else:
        statut_final = statut_academique

    return statut_final, ' · '.join(motifs) if motifs else '—'
