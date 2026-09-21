"""
ZindiClimate — TOP 10 FINAL
Basé sur les résultats réels : AUC=0.841, F1=0.825, Score=0.831

Corrections vs v2 :
  1. XGB lr 0.003 → 0.001  (s'arrêtait à 3-22 arbres, maintenant 300-800)
  2. Blend repondéré : XGB 30%→10%, LGBM 30%→50% (LGBM est le meilleur)
  3. CAT poids maintenu à 30%, ET réduit à 10%
  4. Pseudo-labeling seuil encore plus strict : 0.88/0.12 (plus de qualité)
"""

import pandas as pd
import numpy as np
import os
import warnings
import joblib
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


def load_all(base='data/raw', proc='data/processed'):
    train_raw = pd.read_csv(f'{base}/Train.csv')
    test_raw  = pd.read_csv(f'{base}/Test.csv')
    climate   = pd.read_csv(f'{base}/climate_features.csv')
    ext       = pd.read_csv(f'{proc}/external_data.csv')
    train = train_raw.merge(climate.drop(columns=['deathdate']), on='ID', how='left')
    test  = test_raw.merge(climate.drop(columns=['deathdate']), on='ID', how='left')
    return train, test, test_raw, ext


def impute_ext_data(train, test, ext):
    for col in ['ext_humidity', 'ext_pressure']:
        knn = KNeighborsRegressor(n_neighbors=min(3, len(ext)), weights='distance')
        knn.fit(ext[['latitude','longitude']].values, ext[col].values)
        train[col] = knn.predict(train[['latitude','longitude']].values)
        test[col]  = knn.predict(test[['latitude','longitude']].values)
    return train, test


def add_geo_cluster(train, test, n_clusters=8):
    all_coords = pd.concat([
        train[['latitude','longitude']], test[['latitude','longitude']]
    ]).drop_duplicates().values
    sc = StandardScaler()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(sc.fit_transform(all_coords))
    train['geo_cluster'] = km.predict(sc.transform(train[['latitude','longitude']].values))
    test['geo_cluster']  = km.predict(sc.transform(test[['latitude','longitude']].values))
    return train, test


def build_features(df):
    df = df.copy()
    df['deathdate']    = pd.to_datetime(df['deathdate'])
    df['month']        = df['deathdate'].dt.month
    df['year']         = df['deathdate'].dt.year
    df['day_of_year']  = df['deathdate'].dt.dayofyear
    df['season']       = df['month'].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})
    df['gender_enc']   = df['gender'].map({'Male':0,'Female':1})
    df['zone_enc']     = df['zone'].map({'Rural':0,'Peri_urban':1})

    a = df['age']
    df['age_0_1']      = (a<=1).astype(int)
    df['age_0_2']      = (a<=2).astype(int)
    df['age_0_3']      = (a<=3).astype(int)
    df['age_0_5']      = (a<=5).astype(int)
    df['age_0_10']     = (a<=10).astype(int)
    df['age_0_15']     = (a<=15).astype(int)
    df['age_0_20']     = (a<=20).astype(int)
    df['age_50p']      = (a>=50).astype(int)
    df['age_60p']      = (a>=60).astype(int)
    df['age_65p']      = (a>=65).astype(int)
    df['age_sq']       = a**2
    df['age_cb']       = a**3
    df['age_log']      = np.log1p(a)
    df['age_sqrt']     = np.sqrt(a)
    df['age_inv']      = 1.0/(a+1.0)
    df['age_inv_sq']   = df['age_inv']**2
    df['age_bucket']   = pd.cut(a,
        bins=[-1,1,3,5,10,15,20,30,40,50,60,70,200],
        labels=range(12)).astype(float)

    df['age_inv_x_year']  = df['age_inv'] * df['year']
    df['age_0_5_x_year']  = df['age_0_5'] * df['year']
    df['age_0_3_x_year']  = df['age_0_3'] * df['year']
    df['year_sq']         = df['year']**2
    df['year_norm']       = (df['year'] - 2007) / (2022 - 2007)

    df['temp_range_30d']    = df['tmax_30d'] - df['tmin_30d']
    df['temp_diff_avg_max'] = df['tmax_30d'] - df['tavg_30d']
    df['temp_trend_30_90']  = df['tavg_30d'] - df['tavg_90d']
    df['temp_trend_7_30']   = df['tavg_7d']  - df['tavg_30d']
    df['heat_stress']       = (df['tmax_30d']>35).astype(int)
    df['temp_x_humidity']   = df['tavg_30d'] * df['ext_humidity']
    df['heat_index']        = df['tmax_30d'] * df['ext_humidity'] / 100

    df['rain_trend_30_90']  = df['rain_sum_30d'] - df['rain_sum_90d']/3
    df['rain_frac_7_30']    = df['rain_sum_7d'] / (df['rain_sum_30d']+1)
    df['rain_intensity']    = df['max_daily_rain_30d'] / (df['rain_days_30d']+1)
    df['rain_days_frac']    = df['rain_days_30d']/30.0
    df['rain_sq']           = df['rain_sum_30d']**2
    df['log_rain']          = np.log1p(df['rain_sum_30d'])

    df['ndvi_trend']        = df['ndvi_30d'] - df['ndvi_90d']
    df['drought_index']     = df['ndvi_30d'] / (df['rain_sum_30d']+1)
    df['ndvi_rain']         = df['ndvi_30d'] * df['rain_sum_30d']

    df['pressure_anom']     = df['ext_pressure'] - df['ext_pressure'].median()
    df['humidity_sq']       = df['ext_humidity']**2
    df['humid_x_rain']      = df['ext_humidity'] * df['rain_sum_30d']
    df['lat_lon']           = df['latitude'] * df['longitude']
    df['elevation_sq']      = df['elevation']**2
    df['elev_per_slope']    = df['elevation'] / (df['slope']+0.01)

    df['age_inv_x_rain']    = df['age_inv'] * df['rain_sum_30d']
    df['age_inv_x_temp']    = df['age_inv'] * df['tmax_30d']
    df['age_inv_x_ndvi']    = df['age_inv'] * df['ndvi_30d']
    df['age_inv_x_humid']   = df['age_inv'] * df['ext_humidity']
    df['age_0_5_x_rain']    = df['age_0_5'] * df['rain_sum_30d']
    df['age_0_5_x_temp']    = df['age_0_5'] * df['tavg_30d']
    df['age_0_3_x_rain']    = df['age_0_3'] * df['rain_sum_30d']
    df['age_65p_x_heat']    = df['age_65p'] * df['tmax_30d']
    df['age_x_month']       = df['age_inv'] * df['month']
    df['age_inv_x_log_rain']= df['age_inv'] * df['log_rain']
    return df


BASE_FEATURES = [
    'age','age_sq','age_cb','age_log','age_sqrt','age_inv','age_inv_sq',
    'age_0_1','age_0_2','age_0_3','age_0_5','age_0_10','age_0_15','age_0_20',
    'age_50p','age_60p','age_65p','age_bucket',
    'gender_enc','zone_enc',
    'month','season','year','year_sq','year_norm','day_of_year',
    'age_inv_x_year','age_0_5_x_year','age_0_3_x_year',
    'latitude','longitude','lat_lon','elevation','elevation_sq','slope','elev_per_slope',
    'avg_temperature','max_temperature','min_temperature',
    'tavg_7d','tavg_30d','tavg_90d','tmax_30d','tmin_30d','hot_days_30d',
    'temp_range_mean_30d','temp_range_30d','temp_diff_avg_max',
    'temp_trend_30_90','temp_trend_7_30','heat_stress',
    'precipitation',
    'rain_sum_7d','rain_sum_30d','rain_sum_90d','rain_days_30d','max_daily_rain_30d',
    'rain_trend_30_90','rain_frac_7_30','rain_intensity','rain_days_frac','rain_sq','log_rain',
    'ndvi_30d','ndvi_90d','ndvi_trend','drought_index','ndvi_rain',
    'ext_humidity','ext_pressure','pressure_anom','humidity_sq',
    'temp_x_humidity','heat_index','humid_x_rain',
    'age_inv_x_rain','age_inv_x_temp','age_inv_x_ndvi','age_inv_x_humid',
    'age_0_5_x_rain','age_0_5_x_temp','age_0_3_x_rain',
    'age_65p_x_heat','age_x_month','age_inv_x_log_rain',
    'zone_risk','geo_cluster_risk','year_risk',
]


def target_encode(train_df, val_df, test_df, col, target_series, smoothing=15):
    gm = target_series.mean()
    stats = train_df.groupby(col)[target_series.name].agg(['mean','count'])
    smooth = (stats['count']*stats['mean'] + smoothing*gm) / (stats['count']+smoothing)
    d = smooth.to_dict()
    return (train_df[col].map(d).fillna(gm),
            val_df[col].map(d).fillna(gm),
            test_df[col].map(d).fillna(gm))


def calibrate_probas(oof_probs, y_true, test_probs):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_probs, y_true)
    oof_cal  = iso.predict(oof_probs)
    test_cal = iso.predict(test_probs)
    auc_b = roc_auc_score(y_true, oof_probs)
    auc_a = roc_auc_score(y_true, oof_cal)
    print(f"   Calibration : AUC {auc_b:.4f} → {auc_a:.4f}  (delta={auc_a-auc_b:+.4f})")
    print(f"   q10: {np.quantile(oof_probs,0.1):.3f}→{np.quantile(oof_cal,0.1):.3f}  "
          f"q90: {np.quantile(oof_probs,0.9):.3f}→{np.quantile(oof_cal,0.9):.3f}")
    if auc_a < auc_b - 0.001:
        print("   → Dégradation détectée, calibration annulée")
        return oof_probs, test_probs
    return oof_cal, test_cal


def find_best_threshold(y, oof_probs):
    """Optimise directement 0.6*F1 + 0.4*AUC avec contrainte sur le taux de positifs."""
    auc = roc_auc_score(y, oof_probs)
    best_thr, best_score, best_f1 = 0.5, 0, 0
    for thr in np.arange(0.20, 0.80, 0.003):
        preds = (oof_probs >= thr).astype(int)
        pos   = preds.mean()
        if pos > 0.72 or pos < 0.50:
            continue
        f1    = f1_score(y, preds)
        score = 0.6*f1 + 0.4*auc
        if score > best_score:
            best_score, best_thr, best_f1 = score, thr, f1
    return best_thr, best_score, best_f1


def run_pipeline(train, test, n_splits=5):
    train_fe = build_features(train)
    test_fe  = build_features(test)
    y   = train_fe['is_climate_sensitive']
    spw = float((y==0).sum()) / float((y==1).sum())
    print(f"⚖️  Positifs: {(y==1).sum()} ({y.mean():.1%})  spw={spw:.2f}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_xgb=np.zeros(len(train_fe)); oof_lgbm=np.zeros(len(train_fe))
    oof_cat=np.zeros(len(train_fe)); oof_et=np.zeros(len(train_fe))
    test_xgb=np.zeros(len(test_fe)); test_lgbm=np.zeros(len(test_fe))
    test_cat=np.zeros(len(test_fe)); test_et=np.zeros(len(test_fe))

    print(f"\n🤖 {n_splits}-Fold × 4 modèles...\n")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_fe, y), 1):
        X_tr_r  = train_fe.iloc[tr_idx].copy()
        X_val_r = train_fe.iloc[val_idx].copy()
        y_tr    = y.iloc[tr_idx].rename('is_climate_sensitive')
        y_val   = y.iloc[val_idx]

        # Target encodings fold-aware
        zr_tr,zr_val,zr_ts = target_encode(X_tr_r,X_val_r,test_fe,'zone',y_tr,15)
        cr_tr,cr_val,cr_ts = target_encode(X_tr_r,X_val_r,test_fe,'geo_cluster',y_tr,10)
        yr_tr,yr_val,yr_ts = target_encode(X_tr_r,X_val_r,test_fe,'year',y_tr,20)
        X_tr_r['zone_risk']=zr_tr;         X_val_r['zone_risk']=zr_val
        X_tr_r['geo_cluster_risk']=cr_tr;  X_val_r['geo_cluster_risk']=cr_val
        X_tr_r['year_risk']=yr_tr;         X_val_r['year_risk']=yr_val
        tf=test_fe.copy()
        tf['zone_risk']=zr_ts; tf['geo_cluster_risk']=cr_ts; tf['year_risk']=yr_ts

        available = [f for f in BASE_FEATURES if f in X_tr_r.columns]
        med  = X_tr_r[available].median()
        X_tr = X_tr_r[available].fillna(med)
        X_val= X_val_r[available].fillna(med)
        X_ts = tf[available].fillna(med)

        # ── CORRECTION 1 : XGB stabilisé lr=0.001 ─────────────────
        m1 = XGBClassifier(
            n_estimators=10000, learning_rate=0.001,   # ← 0.003 → 0.001
            max_depth=6, subsample=0.8,
            colsample_bytree=0.75, colsample_bylevel=0.75,
            min_child_weight=15,                        # ← 10 → 15
            gamma=0.3, reg_alpha=0.3, reg_lambda=4.0,
            scale_pos_weight=spw, eval_metric='auc',
            early_stopping_rounds=200,                  # ← 150 → 200
            random_state=42, tree_method='hist', verbosity=0
        )
        # ── LGBM : meilleur modèle → paramètres boostés ───────────
        m2 = LGBMClassifier(
            n_estimators=5000, learning_rate=0.004,    # ← 0.005 → 0.004
            num_leaves=95,                              # ← 63 → 95
            max_depth=8,                                # ← 7 → 8
            subsample=0.8, colsample_bytree=0.75,
            min_child_samples=15,                       # ← 20 → 15
            reg_alpha=0.05, reg_lambda=1.0,
            class_weight='balanced', verbosity=-1, random_state=42
        )
        m3 = CatBoostClassifier(
            iterations=3000, learning_rate=0.01, depth=6,
            l2_leaf_reg=5, auto_class_weights='Balanced',
            eval_metric='AUC', early_stopping_rounds=150,
            random_seed=42, verbose=0
        )
        m4 = ExtraTreesClassifier(
            n_estimators=800, max_depth=None, min_samples_leaf=3,
            class_weight='balanced', random_state=42, n_jobs=-1
        )

        m1.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], verbose=False)
        m2.fit(X_tr, y_tr,
               eval_set=[(X_val,y_val)],
               callbacks=[lgb.early_stopping(150,verbose=False),lgb.log_evaluation(-1)])
        m3.fit(X_tr, y_tr, eval_set=(X_val,y_val))
        m4.fit(X_tr, y_tr)

        oof_xgb[val_idx]  = m1.predict_proba(X_val)[:,1]
        oof_lgbm[val_idx] = m2.predict_proba(X_val)[:,1]
        oof_cat[val_idx]  = m3.predict_proba(X_val)[:,1]
        oof_et[val_idx]   = m4.predict_proba(X_val)[:,1]
        test_xgb  += m1.predict_proba(X_ts)[:,1] / n_splits
        test_lgbm += m2.predict_proba(X_ts)[:,1] / n_splits
        test_cat  += m3.predict_proba(X_ts)[:,1] / n_splits
        test_et   += m4.predict_proba(X_ts)[:,1] / n_splits

        # CORRECTION 2 : blend repondéré — LGBM 50%, CAT 30%, XGB 10%, ET 10%
        p_val = (0.10*oof_xgb[val_idx] + 0.50*oof_lgbm[val_idx] +
                 0.30*oof_cat[val_idx]  + 0.10*oof_et[val_idx])
        print(f"  Fold {fold}/{n_splits} → F1={f1_score(y_val,(p_val>=0.5).astype(int)):.4f}  "
              f"AUC={roc_auc_score(y_val,p_val):.4f}  "
              f"XGB={m1.best_iteration}  LGBM={m2.best_iteration_}  CAT={m3.best_iteration_}")

    # ── CORRECTION 2 : blend repondéré sur tout le test ───────────
    oof_blend  = (0.10*oof_xgb + 0.50*oof_lgbm + 0.30*oof_cat + 0.10*oof_et)
    test_blend = (0.10*test_xgb + 0.50*test_lgbm + 0.30*test_cat + 0.10*test_et)
    auc_blend  = roc_auc_score(y, oof_blend)
    print(f"\n📊 AUC OOF blend = {auc_blend:.4f}")

    # StackNet niveau 2
    print("🏗️  StackNet niveau 2...")
    meta_train = np.column_stack([oof_xgb,oof_lgbm,oof_cat,oof_et])
    meta_test  = np.column_stack([test_xgb,test_lgbm,test_cat,test_et])
    meta_oof=np.zeros(len(y)); meta_test_p=np.zeros(len(test_fe))
    for _,(tr2,val2) in enumerate(StratifiedKFold(5,shuffle=True,random_state=123).split(meta_train,y)):
        lr=LogisticRegression(C=1.0,class_weight='balanced',max_iter=1000)
        lr.fit(meta_train[tr2], y.iloc[tr2])
        meta_oof[val2] = lr.predict_proba(meta_train[val2])[:,1]
        meta_test_p   += lr.predict_proba(meta_test)[:,1] / 5

    auc_meta = roc_auc_score(y, meta_oof)
    print(f"   Stack AUC={auc_meta:.4f}  Blend AUC={auc_blend:.4f}  delta={auc_meta-auc_blend:+.4f}")
    if auc_meta > auc_blend:
        oof_final  = 0.70*meta_oof  + 0.30*oof_blend
        test_final = 0.70*meta_test_p + 0.30*test_blend
        print("   → Stack retenu")
    else:
        oof_final  = oof_blend
        test_final = test_blend
        print("   → Blend direct retenu")

    # Calibration isotonic
    print("🎯 Calibration Isotonic...")
    oof_cal, test_cal = calibrate_probas(oof_final, y.values, test_final)
    return oof_cal, test_cal, y


if __name__ == "__main__":
    BASE = 'data/raw'
    PROC = 'data/processed'

    print("📂 Chargement...")
    train, test, test_raw, ext = load_all(BASE, PROC)

    print("🌍 Imputation KNN ext_data...")
    train, test = impute_ext_data(train, test, ext)

    print("🗺️  Clustering géographique...")
    train, test = add_geo_cluster(train, test, n_clusters=8)

    # ── ROUND 1 ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print("ROUND 1 — Entraînement principal")
    print("="*65)
    oof1, test1, y = run_pipeline(train, test, n_splits=5)
    thr1, score1, f1_1 = find_best_threshold(y, oof1)
    auc1 = roc_auc_score(y, oof1)
    print(f"\n📊 Round 1 → F1={f1_1:.4f}  AUC={auc1:.4f}  Score={score1:.4f}  "
          f"Seuil={thr1:.3f}  Positifs={(oof1>=thr1).mean():.1%}")

    # ── ROUND 2 : Pseudo-labeling ─────────────────────────────────
    print("\n" + "="*65)
    print("ROUND 2 — Pseudo-labeling")
    print("="*65)
    # Seuil plus strict : 0.88/0.12 → plus de qualité, moins de bruit
    conf_high, conf_low = 0.88, 0.12
    high_conf = (test1 >= conf_high) | (test1 <= conf_low)
    n_pseudo  = high_conf.sum()
    print(f"🔄 {n_pseudo} exemples confiants (seuil {conf_high}/{conf_low})")

    if n_pseudo >= 30:
        pseudo_test = test[high_conf].copy()
        pseudo_test['is_climate_sensitive'] = (test1[high_conf] >= 0.5).astype(int)
        train_aug = pd.concat([train, pseudo_test], ignore_index=True)
        train_aug, test_aug = impute_ext_data(train_aug, test.copy(), ext)
        train_aug, test_aug = add_geo_cluster(train_aug, test_aug, n_clusters=8)
        print(f"   Train augmenté : {len(train_aug)} lignes (+{n_pseudo} pseudo)")
        oof2, test2, _ = run_pipeline(train_aug, test_aug, n_splits=5)

        # Blend conservateur R2 : 80% R1 + 20% R2
        test_blend_final = 0.80 * test1 + 0.20 * test2

        # Recalibration du blend final avec isotonic de R1
        print("🎯 Recalibration blend final...")
        iso_final = IsotonicRegression(out_of_bounds='clip')
        iso_final.fit(oof1, y.values)
        test_final = iso_final.predict(test_blend_final)
        oof_final  = oof1
        print(f"   Blend: 80% R1 + 20% R2  |  positifs test: {(test_final>=thr1).mean():.1%}")
    else:
        print("   → Trop peu d'exemples, skip")
        test_final = test1
        oof_final  = oof1

    # ── RÉSULTATS FINAUX ─────────────────────────────────────────
    best_thr, best_score, best_f1 = find_best_threshold(y, oof_final)
    best_auc = roc_auc_score(y, oof_final)

    print(f"\n{'='*65}")
    print(f"✅ OOF F1        : {best_f1:.4f}")
    print(f"✅ OOF AUC-ROC   : {best_auc:.4f}")
    print(f"🚀 SCORE ESTIMÉ  : {best_score:.4f}")
    print(f"🎯 Seuil optimal : {best_thr:.3f}")
    print(f"   Positifs test : {(test_final>=best_thr).mean():.1%}  ← cible 60-70%")
    print(f"{'='*65}")

    # ── SOUMISSION ────────────────────────────────────────────────
    os.makedirs('submissions', exist_ok=True)
    pd.DataFrame({
        'ID':         test_raw['ID'],
        'TargetF1':   (test_final >= best_thr).astype(int),
        'TargetRAUC': test_final
    }).to_csv('submissions/submission_top10_v3.csv', index=False)
    print("\n📁 'submissions/submission_top10_v3.csv' prêt !")

    print("\n📈 Distribution probas test :")
    for thr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print(f"   >= {thr:.1f} : {(test_final>=thr).mean():.1%}")
    print(f"\n   std={test_final.std():.4f}  "
          f"q10={np.quantile(test_final,0.1):.4f}  "
          f"q90={np.quantile(test_final,0.9):.4f}")

    # ── SAUVEGARDE ARTEFACTS POUR INFÉRENCE HF ───────────────────
    print("\n💾 Sauvegarde des artefacts pour déploiement...")
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    train_fe = build_features(train)
    test_fe  = build_features(test)
    available = [f for f in BASE_FEATURES if f in train_fe.columns]
    feature_medians = train_fe[available].median()

    knn_imputers = {}
    for col in ['ext_humidity', 'ext_pressure']:
        knn = KNeighborsRegressor(n_neighbors=min(3, len(ext)), weights='distance')
        knn.fit(ext[['latitude','longitude']].values, ext[col].values)
        knn_imputers[col] = knn

    all_coords = pd.concat([
        train[['latitude','longitude']], test[['latitude','longitude']]
    ]).drop_duplicates().values
    scaler_geo = StandardScaler()
    kmeans_geo = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans_geo.fit(scaler_geo.fit_transform(all_coords))

    te_zone = target_encode(train_fe, train_fe, test_fe, 'zone', y, 15)[0]
    te_geo  = target_encode(train_fe, train_fe, test_fe, 'geo_cluster', y, 10)[0]
    te_year = target_encode(train_fe, train_fe, test_fe, 'year', y, 20)[0]
    target_encoders = {
        'zone': dict(zip(train_fe['zone'], te_zone)),
        'geo_cluster': dict(zip(train_fe['geo_cluster'], te_geo)),
        'year': dict(zip(train_fe['year'], te_year)),
    }

    train_fe['zone_risk'] = te_zone
    train_fe['geo_cluster_risk'] = te_geo
    train_fe['year_risk'] = te_year
    global_mean = y.mean()

    # Réentraîner les modèles finaux sur tout le train
    print("🤖 Réentraînement final sur train complet...")
    X_all = train_fe[available].fillna(feature_medians)
    spw = float((y==0).sum()) / float((y==1).sum())

    m1 = XGBClassifier(
        n_estimators=10000, learning_rate=0.001,
        max_depth=6, subsample=0.8,
        colsample_bytree=0.75, colsample_bylevel=0.75,
        min_child_weight=15, gamma=0.3, reg_alpha=0.3, reg_lambda=4.0,
        scale_pos_weight=spw, eval_metric='auc',
        early_stopping_rounds=200, random_state=42,
        tree_method='hist', verbosity=0
    )
    m2 = LGBMClassifier(
        n_estimators=5000, learning_rate=0.004,
        num_leaves=95, max_depth=8,
        subsample=0.8, colsample_bytree=0.75,
        min_child_samples=15, reg_alpha=0.05, reg_lambda=1.0,
        class_weight='balanced', verbosity=-1, random_state=42
    )
    m3 = CatBoostClassifier(
        iterations=3000, learning_rate=0.01, depth=6,
        l2_leaf_reg=5, auto_class_weights='Balanced',
        eval_metric='AUC', early_stopping_rounds=150,
        random_seed=42, verbose=0
    )
    m4 = ExtraTreesClassifier(
        n_estimators=800, max_depth=None, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    m1.fit(X_all, y, eval_set=[(X_all, y)], verbose=False)
    m2.fit(X_all, y, eval_set=[(X_all, y)],
           callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)])
    m3.fit(X_all, y, eval_set=(X_all, y))
    m4.fit(X_all, y)

    # StackNet sur OOF du train complet (approximation par CV 5-fold)
    meta_train = np.column_stack([
        m1.predict_proba(X_all)[:,1],
        m2.predict_proba(X_all)[:,1],
        m3.predict_proba(X_all)[:,1],
        m4.predict_proba(X_all)[:,1],
    ])
    meta_oof = np.zeros(len(y))
    for _, (tr2, val2) in enumerate(StratifiedKFold(5, shuffle=True, random_state=123).split(meta_train, y)):
        lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
        lr.fit(meta_train[tr2], y.iloc[tr2])
        meta_oof[val2] = lr.predict_proba(meta_train[val2])[:,1]
    auc_blend_save = roc_auc_score(y, oof_final)
    use_stacknet = roc_auc_score(y, meta_oof) > auc_blend_save
    meta_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    meta_model.fit(meta_train, y)

    iso_final = IsotonicRegression(out_of_bounds='clip')
    iso_final.fit(oof_final, y.values)

    artifacts = {
        'models': {
            'xgb': m1, 'lgbm': m2, 'cat': m3, 'et': m4,
        },
        'meta_model': meta_model,
        'iso_calibrator': iso_final,
        'knn_imputer': knn_imputers,
        'kmeans_geo': kmeans_geo,
        'scaler_geo': scaler_geo,
        'target_encoders': target_encoders,
        'feature_medians': feature_medians,
        'BASE_FEATURES': BASE_FEATURES,
        'best_thr': best_thr,
        'global_mean': global_mean,
        'use_stacknet': bool(use_stacknet),
    }
    joblib.dump(artifacts, 'model_artifacts.pkl', compress=3)
    print("✅ Artefacts sauvegardés dans 'model_artifacts.pkl'")