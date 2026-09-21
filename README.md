---
title: "Zindi Climate & Health Predictor"
emoji: "🌍"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🌍 Zindi Climate & Health Predictor

Modèle de prédiction de la sensibilité climatique des décès en Afrique de l'Est.

- **Compétition** : Zindi Climate Change Impact on Mortality
- **Score** : 0.8328 (Rank 41)
- **Architecture** : Stacking + Pseudo-labeling (XGB + LGBM + CAT + ET)

## Utilisation

### Prédiction unitaire
Entrez les informations du cas (âge, localisation, date, climat) et obtenez une probabilité de sensibilité climatique.  
L'application récupère automatiquement l'humidité et la pression via Open-Meteo.

### Prédiction batch
Chargez un CSV au format du test set et téléchargez les prédictions.

## Déploiement

1. Entraîner localement : `python zindiclimateelite.py`
2. Copier `model_artifacts.pkl` dans ce repo
3. Push sur Hugging Face Spaces

## Dependencies

Voir `requirements_hf.txt`
