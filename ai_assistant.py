"""
Module d'assistance IA pour l'aide à la décision utilisant Ollama/Llama3.2
Génère des diagnostics et recommandations basés sur les données d'étudiants
"""

import json
import requests
from typing import Optional, Tuple
import pandas as pd

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"

# Configuration
COLORS = {
    'success': '#4caf50',
    'warning': '#ffa534',
    'danger': '#ff5c5c',
}


def is_ollama_available() -> bool:
    """Vérifie si Ollama est accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except (requests.RequestException, ConnectionError):
        return False


def call_ollama(prompt: str, temperature: float = 0.7) -> str:
    """
    Appelle Ollama pour générer du texte via Llama3.2
    Args:
        prompt: Le prompt à envoyer au modèle
        temperature: Contrôle la créativité (0-1)
    Returns:
        Le texte généré par le modèle
    """
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
        else:
            return None
    except Exception as e:
        print(f"Erreur Ollama: {e}")
        return None


def _diagnostic_alert_level(stats: dict) -> Tuple[str, str]:
    if stats['average'] < 10 or stats['absence_rate'] > 25:
        return 'high', 'danger'
    if stats['average'] < 11 or stats['absence_rate'] > 18:
        return 'medium', 'warning'
    return 'low', 'success'


def _generate_diagnostic_fast(stats: dict, semester: str) -> dict:
    severity, alert_level = _diagnostic_alert_level(stats)
    weak = stats['module_mean'].nsmallest(2).index.tolist() if len(stats.get('module_mean', [])) else []
    modules_text = ', '.join(weak) if weak else 'modules du semestre'
    nlp = stats.get('sentiment_risk_count', 0)
    diagnostic = (
        f"Semestre {semester} : réussite {stats['success_rate']:.0f}%, "
        f"moyenne {stats['average']:.2f}/20, {stats['risk_count']} étudiants en risque."
    )
    if weak:
        diagnostic += f" Modules à surveiller : {modules_text}."
    if nlp:
        diagnostic += f" {nlp} alerte(s) décrochage détectée(s) via les commentaires."
    return {
        'diagnostic': diagnostic,
        'alert_level': alert_level,
        'severity': severity,
        'conclusion': 'Critique' if alert_level == 'danger' else 'Attention' if alert_level == 'warning' else 'Stabilisé',
    }


def _generate_recommendations_fast(stats: dict, semester: str, alert_students: pd.DataFrame) -> list:
    recommendations = []
    weak_modules = stats['module_mean'].nsmallest(2)
    if len(weak_modules) > 0:
        recommendations.append({
            'title': f"Renforcer {', '.join(weak_modules.index.tolist())}",
            'description': f"Moyennes les plus basses du {semester}.",
            'actions': ['Organiser des séances de rattrapage ciblées', 'Proposer des TD supplémentaires'],
            'priority': 'high',
            'status_color': COLORS['warning'],
        })
    if stats.get('sentiment_risk_count', 0) > 0:
        recommendations.append({
            'title': 'Suivi psychologique / décrochage',
            'description': 'Des commentaires tuteur signalent un risque de désengagement.',
            'actions': ['Planifier un entretien individuel', 'Impliquer le référent pédagogique'],
            'priority': 'urgent',
            'status_color': COLORS['danger'],
        })
    if stats['absence_rate'] > 15:
        recommendations.append({
            'title': 'Réduire les absences',
            'description': f"{stats['absence_rate']:.0f}% des étudiants dépassent 18h d'absences.",
            'actions': ['Contacter les étudiants les plus absents', "Vérifier l'emploi du temps"],
            'priority': 'high',
            'status_color': COLORS['warning'],
        })
    if len(alert_students) > 0:
        top = alert_students.iloc[0]
        recommendations.append({
            'title': f"Priorité : {top['Nom']} {top['Prenom']}",
            'description': f"Moyenne {top['Moyenne']:.2f} · {top.get('Motif_alerte', '—')}",
            'actions': ['Convoquer en entretien cette semaine'],
            'priority': 'urgent',
            'status_color': COLORS['danger'],
        })
    if not recommendations:
        recommendations.append({
            'title': 'Performances stables',
            'description': f"Le {semester} ne présente pas d'alerte majeure.",
            'actions': ['Maintenir le suivi régulier'],
            'priority': 'low',
            'status_color': COLORS['success'],
        })
    return recommendations[:3]


def generate_diagnostic(stats: dict, semester: str, use_ollama: bool = False) -> dict:
    """
    Génère un diagnostic IA pour un semestre donné
    
    **Niveau DÉTECTER**: Analyse des KPIs pour identifier les problèmes
    
    Args:
        stats: Dictionnaire contenant les statistiques du semestre
        semester: Code du semestre (S1, S2, etc.)
    
    Returns:
        Dictionnaire avec diagnostic et niveau d'alerte
    """
    if not use_ollama:
        return _generate_diagnostic_fast(stats, semester)

    if not is_ollama_available():
        return {
            'diagnostic': (
                "Ollama n'est pas accessible. Lancez Ollama avec le modele "
                f"{MODEL_NAME} pour generer le diagnostic IA."
            ),
            'alert_level': 'danger',
            'severity': 'high',
            'conclusion': 'Service IA indisponible',
        }

    # Construire le prompt avec les données
    modules_info = "\n".join(
        [f"- {mod}: {val:.1f}/20" for mod, val in stats['module_mean'].items()]
    )
    
    prompt = f"""Tu es un assistant pour la gestion d'une filière d'études supérieures.
Analyse ces données de semestre {semester} et fournis un diagnostic court et clair (max 2 phrases):

DONNÉES DU SEMESTRE:
- Taux de réussite: {stats['success_rate']:.1f}%
- Moyenne générale: {stats['average']:.2f}/20
- Taux d'absence: {stats['absence_rate']:.1f}%
- Étudiants en difficulté: {stats['risk_count']}/{stats['total_students']}

PERFORMANCES PAR MODULE:
{modules_info}

Format ta réponse EXACTEMENT ainsi:
[DIAGNOSTIC]
Ton diagnostic court et professionnel

[CONCLUSION]
Stabilisé | Attention | Critique"""

    response = call_ollama(prompt, temperature=0.5)
    
    if response is None:
        return {
            'diagnostic': 'Erreur de connexion au service IA.',
            'alert_level': 'danger',
            'severity': 'high',
        }

    # Parser la réponse
    lines = response.split('\n')
    diagnostic_text = ''
    conclusion = 'Inconnu'
    
    capturing_diagnostic = False
    for line in lines:
        if '[DIAGNOSTIC]' in line:
            capturing_diagnostic = True
            continue
        elif '[CONCLUSION]' in line:
            capturing_diagnostic = False
            continue
        
        if capturing_diagnostic and line.strip():
            diagnostic_text += line.strip() + ' '
        elif 'Stabilisé' in line or 'Attention' in line or 'Critique' in line:
            conclusion = line.strip()

    # Déterminer le niveau de sévérité
    if stats['average'] < 10 or stats['absence_rate'] > 25:
        severity = 'high'
        alert_level = 'danger'
    elif stats['average'] < 11 or stats['absence_rate'] > 18:
        severity = 'medium'
        alert_level = 'warning'
    else:
        severity = 'low'
        alert_level = 'success'

    return {
        'diagnostic': diagnostic_text.strip() or conclusion,
        'alert_level': alert_level,
        'severity': severity,
        'conclusion': conclusion,
    }


def generate_recommendations(
    stats: dict,
    semester: str,
    alert_students: pd.DataFrame,
    use_ollama: bool = False,
) -> list:
    """
    Génère des recommandations concrètes et actionnables
    
    **Niveaux CIBLER, COMPRENDRE, AGIR**: 
    - CIBLER: Identifie les étudiants prioritaires
    - COMPRENDRE: Explique les causes
    - AGIR: Fournit des actions concrètes
    
    Args:
        stats: Dictionnaire des statistiques du semestre
        semester: Code du semestre
        alert_students: DataFrame des étudiants en alerte
    
    Returns:
        Liste de recommandations avec détails et actions
    """
    if not use_ollama:
        return _generate_recommendations_fast(stats, semester, alert_students)

    if not is_ollama_available():
        return [{
            'title': 'Service Ollama indisponible',
            'description': f"Lancez Ollama avec le modele {MODEL_NAME} pour generer les recommandations IA.",
            'actions': [f"Executer: ollama run {MODEL_NAME}"],
            'priority': 'urgent',
            'status_color': COLORS['danger'],
        }]

    recommendations = []

    # 1. Recommandation sur les modules faibles
    weak_modules = stats['module_mean'].nsmallest(2)
    if len(weak_modules) > 0:
        weak_modules_str = ', '.join(weak_modules.index.tolist())
        
        prompt = f"""Tu es expert pédagogique. Propose UNE recommandation concrète pour améliorer:
Modules faibles: {weak_modules_str} (moyennes: {', '.join([f'{v:.1f}' for v in weak_modules.values])})
Semestre: {semester}

Format EXACTEMENT:
[TITRE] Une ligne courte
[ACTION] Action concrète et immédiate
[DETAIL] Justification courte"""

        response = call_ollama(prompt, temperature=0.6)
        if response:
            recommendations.append(_parse_ai_recommendation(response, 'high'))

    # 2. Recommandation sur les absences si problématique
    if stats['absence_rate'] > 15:
        prompt = f"""Tu es coordinatrice d'une filière. L'absence rate est {stats['absence_rate']:.1f}%.
Donne UNE recommandation concrète pour réduire les absences.

Format EXACTEMENT:
[TITRE] Une ligne courte
[ACTION] Action concrète à prendre maintenant
[DETAIL] Pourquoi c'est important"""

        response = call_ollama(prompt, temperature=0.6)
        if response:
            recommendations.append(_parse_ai_recommendation(response, 'high'))

    # 3. Recommandation pour les étudiants en alerte
    if len(alert_students) > 0:
        top_alert = alert_students.iloc[0] if len(alert_students) > 0 else None
        if top_alert is not None:
            prompt = f"""Tu gères une filière. Recommande UNE action pour:{top_alert['Nom']} {top_alert['Prenom']}
- Moyenne: {top_alert['Moyenne']:.2f}/20
- Absences: {int(top_alert['Absences_heures'])}h
- Statut: {top_alert['Statut']}

Format EXACTEMENT:
[TITRE] Une ligne
[ACTION] Action concrète immédiate
[DETAIL] Justification en 1 phrase"""

            response = call_ollama(prompt, temperature=0.6)
            if response:
                recommendations.append(_parse_ai_recommendation(response, 'urgent'))

    # 4. Recommandation générale (si pas assez de recommandations)
    if len(recommendations) < 2:
        prompt = f"""Tu es directeur d'une filière. Le semestre {semester} a:
- Moyenne: {stats['average']:.2f}/20
- Réussite: {stats['success_rate']:.1f}%
- Absences: {stats['absence_rate']:.1f}%

Donne une recommandation GÉNÉRALE pour maintenir ou améliorer.

Format EXACTEMENT:
[TITRE] Une ligne
[ACTION] Action à prendre
[DETAIL] Brève explication"""

        response = call_ollama(prompt, temperature=0.6)
        if response:
            recommendations.append(_parse_ai_recommendation(response, 'medium'))

    return recommendations if recommendations else [{
        'title': 'Reponse Ollama vide',
        'description': f"Le modele {MODEL_NAME} n'a pas retourne de recommandation exploitable.",
        'actions': ['Verifier que le modele Ollama repond correctement'],
        'priority': 'urgent',
        'status_color': COLORS['danger'],
    }]


def _parse_ai_recommendation(response: str, priority: str) -> dict:
    """Parse la réponse du LLM en format structuré"""
    titre = ''
    action = ''
    detail = ''
    
    lines = response.split('\n')
    current_section = None
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        
        if '[TITRE]' in line:
            current_section = 'titre'
            titre = line_clean.replace('[TITRE]', '').strip()
            # Si le titre est vide, prendre la ligne suivante
            if not titre and i + 1 < len(lines):
                titre = lines[i + 1].strip()
        elif '[ACTION]' in line:
            current_section = 'action'
            action = line_clean.replace('[ACTION]', '').strip()
            if not action and i + 1 < len(lines):
                action = lines[i + 1].strip()
        elif '[DETAIL]' in line:
            current_section = 'detail'
            detail = line_clean.replace('[DETAIL]', '').strip()
            if not detail and i + 1 < len(lines):
                detail = lines[i + 1].strip()
        elif line_clean and not line_clean.startswith('['):
            # Si on n'a pas encore capture le contenu, l'ajouter
            if current_section == 'titre' and titre and not titre.startswith('['):
                pass
            elif current_section == 'action' and action and not action.startswith('['):
                pass
            elif current_section == 'detail' and detail and not detail.startswith('['):
                pass

    # Fallback: si le parsing a échoué, utiliser la première partie de la réponse
    if not titre or titre == 'Recommandation':
        # Prendre la première ligne non-vide
        for line in lines:
            if line.strip() and not line.strip().startswith('['):
                titre = line.strip()[:80]  # Limiter à 80 caractères
                break
    
    if not action:
        # Chercher une ligne avec "→" ou utiliser la deuxième ligne
        for line in lines:
            if '→' in line or 'À' not in line:
                action = line.strip()
                if action and not action.startswith('['):
                    break
        if not action:
            action = 'Suivi personnalisé recommandé'
    
    if not detail:
        detail = titre  # Utiliser le titre comme detail si rien d'autre

    # Déterminer la couleur selon la priorité
    if priority == 'urgent':
        status_color = COLORS['danger']
    elif priority == 'high':
        status_color = COLORS['warning']
    else:
        status_color = COLORS['success']

    return {
        'title': titre.strip() or 'Recommandation',
        'description': detail.strip() or action.strip() or 'Amélioration à considérer',
        'actions': [action.strip()] if action.strip() else ['À définir'],
        'priority': priority,
        'status_color': status_color,
    }


def generate_student_insights(student_row: pd.Series) -> Optional[dict]:
    """
    Génère des insights personnalisés pour un étudiant en alerte
    
    Args:
        student_row: Ligne du DataFrame étudiant
    
    Returns:
        Dictionnaire avec insights et recommandations
    """
    if not is_ollama_available():
        return None

    prompt = f"""Analyse cet étudiant en difficulté:
Nom: {student_row['Nom']} {student_row['Prenom']}
Moyenne: {student_row['Moyenne']:.2f}/20
Absences: {int(student_row['Absences_heures'])}h
Statut: {student_row['Statut']}

Donne UNE recommandation CONCRÈTE et IMMÉDIATE pour l'aider.

Format:
[ACTION] L'action à prendre
[JUSTIFICATION] Pourquoi"""

    response = call_ollama(prompt, temperature=0.5)
    if response:
        lines = response.split('\n')
        action = ''
        justif = ''
        for line in lines:
            if '[ACTION]' in line:
                action = line.replace('[ACTION]', '').strip()
            elif '[JUSTIFICATION]' in line:
                justif = line.replace('[JUSTIFICATION]', '').strip()
        
        return {
            'action': action or 'Suivi personnalisé recommandé',
            'justification': justif or 'À évaluer avec l\'étudiant',
        }
    return None
