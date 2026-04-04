# 🌍 Zindi Climate Change & Mortality Challenge (East Africa)

[![Zindi Score](https://img.shields.io/badge/Zindi-Score_0.8328-gold)](https://zindi.africa/competitions/climate-change-impact-on-mortality)
[![Rank](https://img.shields.io/badge/Rank-14-blue)](#)
[![Team](https://img.shields.io/badge/Team-ITN_AI-green)](#)
[![Country](https://img.shields.io/badge/🇧🇮-Burundi-red)](#)

Ce repository contient la **Version 3 (Optimisée)** de ma solution pour la compétition Zindi. Cette version implémente un pipeline de Stacking avancé et une stratégie de pseudo-labeling pour capturer les relations complexes entre anomalies climatiques et mortalité.

---

## 📈 Progression des scores

| Version | Score Zindi | AUC | F1 Score | Rang | Changement clé |
|:---:|:---:|:---:|:---:|:---:|:---|
| Baseline | 0.7819 | — | — | 82+ | XGB + LGBM simples |
| v1 | 0.8114 | 0.7826 | 0.8305 | 82 | Feature engineering basique |
| v2 | 0.8317 | 0.8414 | 0.8252 | 43 | Calibration isotonic + year_risk |
| **v3** | **0.8328** | **0.8411** | **0.8273** | **41** | Blend repondéré + XGB stabilisé |

> Score Zindi = **0.6 × F1 + 0.4 × AUC**

---

## 🚀 Performances Finales (v3)

- **Score Combiné :** 0.83281555
- **F1-Score :** 0.827272727
- **AUC-ROC :** 0.841129785
- **Rang actuel :** 41 / ~200+ équipes
- **Cible Positifs Test :** ~60–70% (via seuil optimal dynamique)

---

## 🔍 Découvertes clés sur les données

Ces insights ont été déterminants pour progresser du rang 82 au rang 41 :

| Signal | Corrélation | Découverte |
|:---|:---:|:---|
| `age` | r = **+0.443** | Signal dominant — age 1–3 ans → **93.6% de positifs !** |
| `year` | r = **-0.174** | 2ème signal — ignoré au départ. 2007–2012 : 73.6% positifs vs 57.7% post-2012 |
| `max_daily_rain_30d` | r = 0.056 | Meilleur signal climatique |
| `ndvi_90d` | r = 0.045 | Végétation à 90 jours |
| `ext_humidity/pressure` | ~0 | **Couvraient 0% du test** → imputation KNN nécessaire |

> **Découverte critique :** `age_inv = 1/(age+1)` est la transformation la plus puissante.
> Elle capture la non-linéarité : un enfant de 2 ans est radicalement différent d'un adulte de 30 ans.

---

## 🛠️ Architecture du Pipeline "Elite v3"

Ma solution repose sur une architecture de **Stacking à deux niveaux** avec enrichissement de données.

### 0. Préparation des données

**Problème découvert :** `external_data.csv` (humidité, pression Open-Meteo) couvrait **0% des coordonnées du test** — le modèle voyait ces features à l'entraînement mais pas à la prédiction.

**Solution :** Imputation `KNeighborsRegressor` (k=3, pondéré par distance) sur la latitude/longitude.

```python
knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
knn.fit(ext[['latitude', 'longitude']].values, ext['ext_humidity'].values)
test['ext_humidity'] = knn.predict(test[['latitude', 'longitude']].values)
```

### 1. Feature Engineering Avancé (70+ features)

#### Age — signal dominant (r = 0.443)
```python
age_inv      = 1 / (age + 1)          # Transformation clé
age_inv_sq   = age_inv ** 2
age_0_3      = (age <= 3)              # Taux cible 93.6% !
age_bucket   = pd.cut(age, 12 bins)   # Granularité fine
```

#### Year — signal ignoré au départ (r = -0.174)
```python
year_risk       = target_encoding(year, smoothing=20)  # Fold-aware
age_inv_x_year  = age_inv * year   # Interaction âge × tendance temporelle
age_0_5_x_year  = age_0_5 * year
```

#### Indices climatiques composites
```python
heat_index     = tmax_30d * ext_humidity / 100
drought_index  = ndvi_30d / (rain_sum_30d + 1)
rain_intensity = max_daily_rain_30d / (rain_days_30d + 1)
log_rain       = np.log1p(rain_sum_30d)
```

#### Interactions biologiques (vulnérabilité × climat)
```python
age_inv_x_rain = age_inv * rain_sum_30d
age_inv_x_temp = age_inv * tmax_30d
age_0_3_x_rain = age_0_3 * rain_sum_30d
age_65p_x_heat = age_65p * tmax_30d
```

#### Target Encoding Fold-Aware (sans data leakage)
```python
# Calculé UNIQUEMENT sur le fold train à chaque fold
smooth = (count * mean + smoothing * global_mean) / (count + smoothing)
# Appliqué sur : zone_risk, geo_cluster_risk, year_risk
```

> **Erreur corrigée :** Le target encoding global causait un **data leakage** sévère.
> Correction : encoding recalculé à l'intérieur de chaque fold de la CV.

- **Imputation Géo-Spatiale :** `KNeighborsRegressor` sur latitude/longitude
- **Clustering Géographique :** 8 clusters `KMeans` pour les spécificités régionales
- **Interactions Biologiques :** Features croisées age × climat

### 2. Ensemble Learning (Niveau 1)

Blend pondéré de 4 modèles diversifiés, poids basés sur les performances OOF observées :

| Modèle | Poids | Itérations/fold | Rôle |
|:---|:---:|:---:|:---|
| **LightGBM** | **50%** | 400–1 000 | Modèle dominant, le plus stable |
| **CatBoost** | **30%** | 13–446 | Excellent sur variables catégorielles |
| **XGBoost** | **10%** | 300–800* | Stabilisé avec lr=0.001 |
| **ExtraTrees** | **10%** | 800 (fixe) | Réduction de variance par bagging pur |

> *XGBoost s'arrêtait à **3 arbres** avec lr=0.003 — poids réduit de 30% à 10%, lr abaissé à 0.001.

**Hyperparamètres clés LightGBM :**
```python
LGBMClassifier(
    n_estimators=5000, learning_rate=0.004,
    num_leaves=95, max_depth=8,
    subsample=0.8, colsample_bytree=0.75,
    min_child_samples=15, class_weight='balanced'
)
```

### 3. Méta-Modèle & Calibration (Niveau 2)

- **StackNet :** `LogisticRegression` entraînée sur les probabilités OOF du niveau 1 (5-fold interne). Retenu uniquement si AUC > blend direct.
- **Calibration Isotonique :** Gain observé : **+0.004 à +0.006 AUC** à chaque run.

```
Avant calibration : q10=0.296  q90=0.808  ← probas trop groupées
Après calibration : q10=0.302  q90=0.954  ← bien étalées → meilleure AUC
```

### 4. Pseudo-labeling (Round 2)

Ré-entraînement complet du pipeline avec les prédictions test les plus fiables :

```python
conf_high, conf_low = 0.88, 0.12      # Seuils stricts → qualité maximale
# ~400 exemples ajoutés au train (3 146 → 3 555 lignes)
test_final = 0.80 * test_round1 + 0.20 * test_round2  # Blend conservateur
```

**Résultat :** AUC Round 2 = **0.8496** (+0.030 vs Round 1).

### 5. Seuil Optimal

```python
# Optimise directement le score Zindi = 0.6*F1 + 0.4*AUC
# Contrainte : 50% ≤ taux positifs ≤ 72%
for thr in np.arange(0.20, 0.80, 0.003):
    score = 0.6 * f1_score(y, preds) + 0.4 * roc_auc_score(y, probas)
```

---

## 📁 Structure du Projet

```text
ZINDICLIMATE/
│
├── data/
│   ├── raw/
│   │   ├── Train.csv                # 3 146 enregistrements de mortalité
│   │   ├── Test.csv                 # 1 030 enregistrements à prédire
│   │   ├── climate_features.csv     # Features ERA5, CHIRPS, MODIS, SRTM
│   │   └── data_dictionary.csv      # Description des variables
│   │
│   └── processed/
│       ├── train_final.csv          # Train fusionné + features engineerées
│       ├── test_final.csv           # Test fusionné + features engineerées
│       └── external_data.csv        # Humidité/pression Open-Meteo (29 coords)
│
├── submissions/
│   ├── submission_top10_v2.csv      # Score 0.8317 — rang 43
│   └── submission_top10_v3.csv      # Score 0.8328 — rang 41 ← meilleure
│
├── zindiclimate_top10_v3.py         # Script principal ← lancer celui-ci
├── get_external_weather.py          # Récupération données Open-Meteo
├── requirements.txt
└── README.md
```

---

## ⚙️ Reproduction des résultats

```bash
# 1. Installer les dépendances
pip install xgboost lightgbm catboost scikit-learn pandas numpy

# 2. Placer les fichiers Zindi dans data/raw/
#    Train.csv, Test.csv, climate_features.csv

# 3. (Optionnel) Récupérer les données météo externes
python get_external_weather.py

# 4. Lancer le pipeline complet (~20-30 min)
python zindiclimate_top10_v3.py

# Output : submissions/submission_top10_v3.csv
```

---

## 🔬 Sources des données

| Source | Variables | Période |
|:---|:---|:---:|
| ERA5-Land (ECMWF) | Température avg/max/min sur 7/30/90j, jours chauds | 2007–2022 |
| CHIRPS Daily | Précipitations sur 7/30/90j, jours de pluie, intensité max | 2007–2022 |
| MODIS MOD13Q1 | NDVI (végétation) sur 30/90j | 2007–2022 |
| SRTM (NASA) | Élévation, pente du terrain | — |
| Open-Meteo Archive | Humidité relative 2m, pression atmosphérique | 2021–2024 |

---

## 🐛 Bugs corrigés — leçons apprises

| Problème | Impact | Solution appliquée |
|:---|:---:|:---|
| Data leakage target encoding | -0.015 AUC | Encoding recalculé à l'intérieur de chaque fold |
| ext_data couvre 0% du test | -0.010 AUC | Imputation KNN par coordonnées GPS |
| `year` ignoré comme feature | -0.015 Score | `year_risk` + interactions `age×year` |
| Seuil cherché entre 0.5–0.6 | -0.008 F1 | Recherche sur 0.20–0.80 avec contrainte |
| XGB s'arrêtait à 3 arbres | -0.005 AUC | lr : 0.01 → 0.001, poids : 30% → 10% |
| Probas trop groupées (AUC faible) | -0.005 AUC | Calibration Isotonic Regression |
| scale_pos_weight fixe à 1.5 | — | Calculé dynamiquement : `sum(y==0)/sum(y==1)` |

---

## 👥 Équipe

**ITN AI** — Équipe Burundaise 🇧🇮

---

*Dernière mise à jour : Mars 2026 · Score : 0.8328 · Rang : 14*
