"""
Pré-calcule les scores de sentiment à partir de data/commentaires_suivi.csv
Usage: python scripts/compute_sentiment.py [--force]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentiment_ml import (
    SCORES_FILE,
    clear_sentiment_cache,
    compute_all_scores,
    ensure_sentiment_scores,
    get_model_label,
    load_comments,
    save_scores,
    _get_sentiment_pipeline,
)


def main() -> None:
    force = '--force' in sys.argv
    if force and SCORES_FILE.exists():
        SCORES_FILE.unlink()

    comments = load_comments()
    if comments.empty:
        print('Aucun commentaire trouvé dans data/commentaires_suivi.csv')
        return

    _get_sentiment_pipeline()
    print(f'Moteur : {get_model_label()}')

    if force:
        print(f'Analyse de {len(comments)} commentaires...')
        scores = compute_all_scores(comments, use_transformer=True)
        path = save_scores(scores)
    else:
        scores = ensure_sentiment_scores()
        path = SCORES_FILE

    clear_sentiment_cache()
    alerts = int(scores['sentiment_alert'].sum()) if 'sentiment_alert' in scores.columns else 0
    print(f'Fichier généré : {path} ({len(scores)} lignes, {alerts} alertes décrochage)')


if __name__ == '__main__':
    main()
