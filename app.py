import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / 'data'
DATA_FILE = DATA_DIR / 'etudiants_filiere.csv'

FILIERES = ['Informatique', 'Mecanique', 'Electronique', 'Gestion', 'BTP']
SEMESTRES = ['S1', 'S2']

MODULES_PAR_FILIERE = {
    'Informatique': ['Python & ML', 'Bases de donnees', 'Architecture logicielle', 'Maths', 'Anglais'],
    'Mecanique': ['Mecanique des fluides', 'Matieres', 'Conception CAO', 'Maths', 'Anglais'],
    'Electronique': ['Circuits', 'Microcontroleurs', 'Telecoms', 'Maths', 'Anglais'],
    'Gestion': ['Management', 'Finance', 'Marketing', 'Maths', 'Anglais'],
    'BTP': ['Resistance des materiaux', 'Topographie', 'Gestion de chantier', 'Maths', 'Anglais'],
}


@st.cache_data
def load_or_create_dataset() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    rng = np.random.default_rng(42)
    n_students = 180

    prenoms = ['Amine', 'Sara', 'Youssef', 'Nora', 'Hassan', 'Imane', 'Karim', 'Lina', 'Rachid', 'Salma']
    noms = ['El Khatib', 'Bennani', 'Ziani', 'Hamdouch', 'Bouchaib', 'Alaoui', 'Mansouri', 'Othman', 'Chafii', 'Idrissi']
    villes = ['Casablanca', 'Rabat', 'Tanger', 'Marrakech', 'Agadir', 'Fes', 'Tetouan']

    rows = []
    for i in range(n_students):
        filiere = rng.choice(FILIERES)
        semestre = rng.choice(SEMESTRES, p=[0.45, 0.55])
        moyenne_base = {
            'Informatique': 13.5,
            'Mecanique': 12.8,
            'Electronique': 12.2,
            'Gestion': 13.1,
            'BTP': 12.4,
        }[filiere]
        ecart = {
            'Informatique': 3.8,
            'Mecanique': 4.2,
            'Electronique': 4.5,
            'Gestion': 3.5,
            'BTP': 4.8,
        }[filiere]
        absence_moy = {
            'Informatique': 14,
            'Mecanique': 18,
            'Electronique': 16,
            'Gestion': 12,
            'BTP': 20,
        }[filiere]

        absences = int(max(0, rng.normal(absence_moy, 5)))
        modules = MODULES_PAR_FILIERE[filiere]
        notes = [round(min(20, max(4, rng.normal(moyenne_base + 0.3 * idx, ecart))), 2) for idx in range(len(modules))]

        rows.append({
            'student_id': f'ST{i+1000:05d}',
            'prenom': rng.choice(prenoms),
            'nom': rng.choice(noms),
            'ville': rng.choice(villes),
            'filiere': filiere,
            'semestre': semestre,
            'absences_heures': absences,
            'moyenne': round(sum(notes) / len(notes), 2),
            'taux_reussite': 1 if sum(notes) / len(notes) >= 10 else 0,
            'alerte': 1 if absences >= 20 or sum(notes) / len(notes) < 10.5 else 0,
            **{f'module_{j+1}': modules[j] for j in range(len(modules))},
            **{f'note_{j+1}': notes[j] for j in range(len(notes))},
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    return df


@st.cache_data
def build_summary(df: pd.DataFrame, filiere: str, semestre: str) -> str:
    selection = df[(df['filiere'] == filiere) & (df['semestre'] == semestre)]
    if selection.empty:
        return 'Aucune donnée disponible pour cette filière / ce semestre.'

    moyenne_generale = selection['moyenne'].mean()
    taux_reussite = selection['taux_reussite'].mean() * 100
    taux_absence = (selection['absences_heures'] > 18).mean() * 100
    notes_cols = [c for c in selection.columns if c.startswith('note_')]

    module_averages = selection[notes_cols].mean()
    modules_en_difficulte = [selection[f'module_{idx+1}'].iloc[0] for idx, note in enumerate(notes_cols) if module_averages[note] < 11]

    points = []
    if taux_reussite < 70:
        points.append(f'Le taux de réussite est faible à {taux_reussite:.0f}% pour {filiere} {semestre}.')
    else:
        points.append(f'Le taux de réussite reste satisfaisant à {taux_reussite:.0f}%.')

    if taux_absence > 25 or selection['absences_heures'].mean() > 17:
        points.append(f'Le taux d’absences moyen est préoccupant ({selection["absences_heures"].mean():.1f} heures).')
    else:
        points.append('Le taux d’absences reste modéré.')

    if modules_en_difficulte:
        points.append('Modules en difficulté : ' + ', '.join(modules_en_difficulte) + '.')
    else:
        points.append('Aucun module majeur n’est clairement en difficulté.')

    recommandations = [
        'Organiser un soutien ciblé sur les modules les plus faibles.',
        'Contacter les étudiants en alerte pour un plan de suivi personnalisé.',
        'Vérifier l’emploi du temps avant la prochaine évaluation.',
    ]

    return (
        f"Voici les statistiques de la filière {filiere} {semestre} : taux de réussite {taux_reussite:.0f}%, moyenne {moyenne_generale:.1f}, "
        f"taux d'absence élevé pour {taux_absence:.0f}% des étudiants.\n\n"
        + ' '.join(points)
        + '\n\nRecommandations :\n• ' + '\n• '.join(recommandations)
    )


def format_modules(selection: pd.DataFrame) -> pd.DataFrame:
    note_cols = [c for c in selection.columns if c.startswith('note_')]
    module_labels = [selection[f'module_{idx+1}'].iloc[0] for idx, _ in enumerate(note_cols)]
    return pd.DataFrame({
        'module': module_labels,
        'moyenne': [selection[col].mean() for col in note_cols],
    })


def main() -> None:
    st.set_page_config(page_title='Tableau de bord filière', layout='wide')
    st.title('Tableau de bord interactif — Analyse de filière')
    st.markdown(
        'Visualisez les KPIs, identifiez les étudiants en alerte, et simulez une première analyse textuelle pour le chef de filière.'
    )

    df = load_or_create_dataset()
    filiere = st.sidebar.selectbox('Choisir la filière', FILIERES)
    semestre = st.sidebar.radio('Choisir le semestre', SEMESTRES)
    alert_threshold = st.sidebar.slider('Seuil d’absences pour alerte', 5, 40, 18)

    selection = df[(df['filiere'] == filiere) & (df['semestre'] == semestre)]

    col1, col2, col3 = st.columns(3)
    col1.metric('Taux de réussite', f"{selection['taux_reussite'].mean() * 100:.0f}%")
    col2.metric('Moyenne générale', f"{selection['moyenne'].mean():.2f}")
    col3.metric('Absents > seuil', f"{(selection['absences_heures'] > alert_threshold).mean() * 100:.0f}%")

    st.subheader('Moyennes par module')
    module_stats = format_modules(selection)
    fig = px.bar(module_stats, x='module', y='moyenne', text='moyenne', labels={'moyenne': 'Moyenne'}, height=420)
    fig.update_traces(marker_color='indianred', texttemplate='%{text:.1f}')
    fig.update_layout(xaxis_tickangle=-35, yaxis_range=[0, 20])
    st.plotly_chart(fig, use_container_width=True)

    selection = selection.copy()
    selection['score_risque'] = (12 - selection['moyenne']).clip(lower=0) + selection['absences_heures'] * 0.25
    alert_table = selection.sort_values('score_risque', ascending=False)[['student_id', 'prenom', 'nom', 'ville', 'moyenne', 'absences_heures']]
    alert_table.columns = ['ID étudiant', 'Prénom', 'Nom', 'Ville', 'Moyenne', 'Absences (h)']

    st.subheader('Liste des étudiants en alerte')
    st.dataframe(alert_table.head(15), use_container_width=True)

    if st.button('Analyser la filière'):
        summary = build_summary(df, filiere, semestre)
        st.subheader('Analyse automatique (simulation)')
        st.info(summary)

    st.markdown('---')
    st.markdown('### Aperçu des données')
    st.write(selection.head(10))


if __name__ == '__main__':
    main()
