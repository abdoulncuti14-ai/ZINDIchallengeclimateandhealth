---
title: "Zindi Climate & Health Predictor"
emoji: "🌍"
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.30
app_file: streamlit_app.py
pinned: false
license: mit
---

# 🌍 Zindi Climate & Health Predictor

Modèle de prédiction de la sensibilité climatique des décès en Afrique de l'Est.

- **Compétition** : Zindi Climate Change Impact on Mortality
- **Score** : 0.8328 (Rank 41)
- **Architecture** : Stacking + Pseudo-labeling (XGB + LGBM + CAT + ET)

## Déploiement

### Streamlit Community Cloud
1. Aller sur https://streamlit.io/cloud
2. Connecter le repo GitHub `abdoulncuti14-ai/ZINDIchallengeclimateandhealth`
3. Main file : `streamlit_app.py`
4. Cliquer sur **Deploy**

### Hugging Face Spaces
1. Créer un Space avec SDK Gradio
2. Connecter le repo GitHub
3. Utiliser `app.py` comme fichier principal

## Dependencies

- `requirements_streamlit.txt` pour Streamlit Cloud
- `requirements_hf.txt` pour Hugging Face Spaces
