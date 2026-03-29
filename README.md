# 🌍 Zindi Climate Change & Mortality Challenge (East Africa)

[![Zindi Score](https://img.shields.io/badge/Zindi-Score_0.831-gold)](https://zindi.africa/competitions/climate-change-impact-on-mortality)
[![Rank](https://img.shields.io/badge/Rank-TOP_10_FINAL-blue)](#)

Ce repository contient la **Version 3 (Optimisée)** de ma solution pour la compétition Zindi. Cette version implémente un pipeline de Stacking avancé et une stratégie de pseudo-labeling pour capturer les relations complexes entre anomalies climatiques et mortalité.

## 🚀 Performances Finales (v3)
- **Score Combiné :** 0.831
- **F1-Score (OOF) :** 0.825
- **AUC-ROC (OOF) :** 0.841
- **Cible Positifs Test :** ~60-70% (via seuil optimal dynamique)

## 🛠️ Architecture du Pipeline "Elite v3"

Ma solution repose sur une architecture de **Stacking à deux niveaux** avec enrichissement de données :

### 1. Feature Engineering Avancé
- **Imputation Géo-Spatiale :** Utilisation de `KNeighborsRegressor` pour imputer les données climatiques externes basées sur la latitude/longitude.
- **Clustering Géographique :** Segmentation du territoire en 8 clusters via `KMeans` pour capturer les spécificités régionales.
- **Interactions Biologiques :** Création de features croisées (ex: `age_inv_x_temp`, `age_0_5_x_rain`) pour modéliser la vulnérabilité des nourrissons.

### 2. Ensemble Learning (Niveau 1)
Blend pondéré de 4 modèles diversifiés pour maximiser la généralisation :
- **LGBM (50%) :** Modèle dominant, optimisé avec `num_leaves=95` et `learning_rate=0.004`.
- **CatBoost (30%) :** Excellent pour la gestion des variables catégorielles.
- **XGBoost (10%) :** Stabilisé avec un `learning_rate` ultra-faible (0.001) et 300-800 itérations.
- **ExtraTrees (10%) :** Pour la réduction de la variance via le bagging pur.

### 3. Méta-Modèle & Calibration (Niveau 2)
- **StackNet :** Régression Logistique entraînée sur les probabilités OOF du niveau 1.
- **Calibration Isotonique :** Ajustement des probabilités finales pour garantir que les scores reflètent les probabilités réelles de décès climat-sensibles.
- **Pseudo-labeling :** Ré-entraînement du pipeline (Round 2) en intégrant les prédictions test les plus fiables (Seuils stricts : >0.88 ou <0.12).

## 📁 Structure du Projet

```text
├── src/                # Script principal (v3_final_pipeline.py)
├── data/
│   ├── raw/            # Train.csv, Test.csv, climate_features.csv
│   └── processed/      # external_data.csv (données interpolées)
├── submissions/        # submission_top10_v3.csv
├── requirements.txt    # Dépendances (XGBoost, LightGBM, CatBoost, etc.)
└── README.md
