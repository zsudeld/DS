"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 07 – Random Forest                                    ║
║  LECKE 12 – Gradient Boosting (XGBoost, LightGBM, CatBoost) ║
║  LECKE 13 – Klaszterezés (K-Means, DBSCAN, Hierarchikus)    ║
╚══════════════════════════════════════════════════════════════╝
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import warnings; warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import make_classification, make_blobs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score, silhouette_score
    from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN, KMeans
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
except ImportError as _hiba:
    _csomag = str(_hiba).replace("No module named ", "").strip("'")
    _pip_map = {'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'Pillow', 'plotly': 'plotly', 'sns': 'seaborn', 'matplotlib': 'matplotlib', 'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'statsmodels': 'statsmodels', 'networkx': 'networkx', 'xgboost': 'xgboost', 'lightgbm': 'lightgbm', 'catboost': 'catboost', 'optuna': 'optuna', 'flaml': 'flaml', 'mlflow': 'mlflow', 'prophet': 'prophet', 'fastapi': 'fastapi', 'uvicorn': 'uvicorn', 'pydantic': 'pydantic', 'joblib': 'joblib', 'anthropic': 'anthropic', 'openai': 'openai', 'pingouin': 'pingouin', 'tqdm': 'tqdm'}
    _pip = _pip_map.get(_csomag.split(".")[0], _csomag.split(".")[0])
    print(f"\n\033[91m❌  Hiányzó csomag: {_csomag}\033[0m")
    print(f"\033[93m👉  Telepítsd: pip install {_pip}\033[0m")
    print("\033[96m💡  Vagy az összes egyszerre: pip install -r requirements.txt\033[0m\n")
    raise SystemExit(1)
# ────────────────────────────────────────────────────────────────
from pathlib import Path as _Path
OUTPUT_DIR = _Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)




# ════════════════════════════════════════════════════════════
# LECKE 07 – RANDOM FOREST
# ════════════════════════════════════════════════════════════

def random_forest_demo() -> None:
    """Random Forest – ensemble módszer bagging-gel.

    MŰKÖDÉSI ELV:
      1. Bootstrap: n véletlenszerű részhalmazt vesz a tanítóadatból
      2. Minden részhalmaz → döntési fa (de véletlenszerű jellemzőkkel!)
      3. Szavazás: a legtöbb fa által prediktált osztály nyeri

    ELŐNYÖK:
      ✅ Robusztus outlierekre
      ✅ Kevés preprocessing kell (nem kell scaling)
      ✅ Beépített feature importance
      ✅ Párhuzamosítható

    HÁTRÁNYOK:
      ❌ Lassabb mint döntési fa
      ❌ Nehezebben értelmezhető
      ❌ Extrapolációra nem jó (idősor előrejelzés problémás)
    """
    print("=" * 55)
    print("  LECKE 07 – Random Forest")
    print("=" * 55)

    # Adat
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, random_state=42, weights=[0.6, 0.4],
    )
    feature_names = [f"f_{i:02d}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)

    # Alap modell
    rf = RandomForestClassifier(
        n_estimators=200,     # fa-k száma: több = stabilabb, de lassabb
        max_depth=None,       # None = teljes mélységig növi
        min_samples_split=5,  # min. minták egy split-hez
        min_samples_leaf=2,   # min. minták egy levélben
        max_features="sqrt",  # jellemzők száma/fa: √n az osztályozáshoz ajánlott
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,            # párhuzamos futtatás
    )
    rf.fit(X_train, y_train)
    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Feature Importance
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    print(f"\nTop 10 fontos jellemző:\n{importances.head(10).to_string()}")

    plt.figure(figsize=(10, 5))
    importances.head(10).plot(kind="bar", color="#27AE60", alpha=0.8)
    plt.title("Random Forest – Feature Importance (Top 10)")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_rf_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Hyperparaméter-hangolás (RandomizedSearch – gyorsabb mint GridSearch)
    print("\n📊 Hyperparaméter-hangolás (RandomizedSearchCV)...")
    param_grid = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "max_features":    ["sqrt", "log2"],
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=20,               # 20 véletlenszerű kombináció
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)
    print(f"  Legjobb paraméterek: {random_search.best_params_}")
    print(f"  Legjobb ROC-AUC (CV): {random_search.best_score_:.4f}")


# ════════════════════════════════════════════════════════════
# LECKE 12 – GRADIENT BOOSTING
# ════════════════════════════════════════════════════════════

def gbm_demo() -> None:
    """XGBoost, LightGBM, CatBoost összehasonlítás.

    GRADIENT BOOSTING ALAPELVE:
      Random Forest: párhuzamos fák (bagging)
      GBM:           szekvenciális fák (boosting)
        → Minden fa a MARADÉKOT (hibát) tanulja az előzőktől.

    XGBoost:   L1/L2 regularizáció, gyors, GPU-s
    LightGBM:  Leaf-wise növekedés, nagyon gyors nagy adatnál
    CatBoost:  Kategorikus változók beépített kezelése, kevés preprocessing
    """
    print("\n" + "=" * 55)
    print("  LECKE 12 – Gradient Boosting")
    print("=" * 55)

    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    eredmenyek = {}

    # XGBoost
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,      # kis tanulási ráta → több fa kell, de jobb
            max_depth=6,
            subsample=0.8,           # soronkénti mintavételezés
            colsample_bytree=0.8,    # jellemzőnkénti mintavételezés
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
        eredmenyek["XGBoost"] = auc
        print(f"  XGBoost ROC-AUC: {auc:.4f}")
    except ImportError:
        print("  XGBoost nem telepítve: pip install xgboost")

    # LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,           # komplexitás kontrollja (max 2^max_depth)
            min_child_samples=20,    # levélminták min. száma
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train)
        auc = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])
        eredmenyek["LightGBM"] = auc
        print(f"  LightGBM ROC-AUC: {auc:.4f}")
    except ImportError:
        print("  LightGBM nem telepítve: pip install lightgbm")

    # CatBoost
    try:
        from catboost import CatBoostClassifier
        catboost = CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        )
        catboost.fit(X_train, y_train)
        auc = roc_auc_score(y_test, catboost.predict_proba(X_test)[:, 1])
        eredmenyek["CatBoost"] = auc
        print(f"  CatBoost ROC-AUC: {auc:.4f}")
    except ImportError:
        print("  CatBoost nem telepítve: pip install catboost")

    # Összehasonlítás
    if eredmenyek:
        legjobb = max(eredmenyek, key=eredmenyek.get)
        print(f"\n  🏆 Legjobb: {legjobb} ({eredmenyek[legjobb]:.4f})")


# ════════════════════════════════════════════════════════════
# LECKE 13 – KLASZTEREZÉS
# ════════════════════════════════════════════════════════════

def klaszterezés_demo() -> None:
    """K-Means, DBSCAN és Hierarchikus klaszterezés.

    FELÜGYELET NÉLKÜLI TANULÁS – nincs célváltozó!

    K-MEANS:
      + Gyors, skálázható
      - K-t előre kell megadni
      - Csak konvex (gömb alakú) klaszterekre jó

    DBSCAN:
      + Nem kell előre K
      + Zajpontokat felismeri
      + Tetszőleges alakú klaszterek
      - Érzékeny eps és min_samples paraméterekre

    HIERARCHIKUS:
      + Nincs K szükséges
      + Dendrogram vizualizáció
      - O(n³) – lassú nagy adatnál
    """
    print("\n" + "=" * 55)
    print("  LECKE 13 – Klaszterezés")
    print("=" * 55)

    # Szintetikus adathalmaz 4 klaszterrel
    X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── K-Means ──────────────────────────────────────────────
    # Elbow-módszer: Optimális K megkeresése
    inertiak = []
    silhouette_scores = []
    k_range = range(2, 10)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertiak.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    # Elbow és Silhouette vizualizáció
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_range, inertiak, "bo-")
    axes[0].set_xlabel("K (klaszterek száma)")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title("Elbow módszer")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouette_scores, "go-")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette elemzés (magasabb = jobb)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.show()

    # K=4 végső modell
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_km = km4.fit_predict(X_scaled)
    print(f"\n  K-Means (k=4) Silhouette: {silhouette_score(X_scaled, labels_km):.4f}")

    # ── DBSCAN ───────────────────────────────────────────────
    dbscan = DBSCAN(
        eps=0.4,           # szomszédsági sugár – adatspecifikus!
        min_samples=5,     # sűrűségi minimum
    )
    labels_db = dbscan.fit_predict(X_scaled)
    n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
    n_zaj = (labels_db == -1).sum()
    print(f"  DBSCAN: {n_clusters_db} klaszter, {n_zaj} zajpont")
    if n_clusters_db > 1:
        print(f"  DBSCAN Silhouette: {silhouette_score(X_scaled[labels_db != -1], labels_db[labels_db != -1]):.4f}")

    # ── Hierarchikus klaszterezés ────────────────────────────
    Z = linkage(X_scaled[:100], method="ward")  # első 100 pont (gyorsaság)

    plt.figure(figsize=(12, 5))
    dendrogram(Z, no_labels=True, color_threshold=3.0)
    plt.title("Hierarchikus klaszterezés – Dendrogram")
    plt.xlabel("Minták")
    plt.ylabel("Távolság (Ward)")
    plt.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="Vágási szint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Klaszterek vizualizálása egymás mellett
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (labels, cim) in zip(axes, [
        (y_true,    "Valódi klaszterek"),
        (labels_km, "K-Means (k=4)"),
        (labels_db, "DBSCAN"),
    ]):
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
        ax.set_title(cim)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "13_klaszter_osszehasonlitas.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    random_forest_demo()
    gbm_demo()
    klaszterezés_demo()

    print("\n✅ Leckék 07/12/13 sikeresen lefutottak!")
    print("➡️  Következő: 09_ml_automl.py")