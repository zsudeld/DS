"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 17 – Train/Test Split és Cross-Validation             ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - Train/Validation/Test felosztás helyes módja
  - Stratified split (osztályarányok megőrzése)
  - K-Fold, StratifiedKFold, TimeSeriesSplit
  - Data leakage azonosítása és elkerülése
  - Overfitting vs. underfitting diagnosztika

ARANY SZABÁLY:
  A test set CSAK EGYSZER, a végső kiértékelésnél látható!
  Minden optimalizáció a train+validation seten történik.
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
        cross_val_score,
        learning_curve,
        train_test_split,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
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
# 1. ALAP FELOSZTÁS
# ════════════════════════════════════════════════════════════

def alap_felosztás_demo() -> None:
    """Train/Validation/Test 60/20/20 felosztás.

    MIÉRT HÁROM RÉSZ?
      train:      Modell tanítása
      validation: Hyperparaméter-hangolás, modellválasztás
      test:       VÉGSŐ, torzítatlan kiértékelés (SOHA NEM LÁTOD ELŐRE!)
    """
    print("=== 1. ALAP FELOSZTÁS ===")

    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_classes=2, random_state=42, weights=[0.7, 0.3]  # kiegyenlítetlen osztályok
    )

    # 1. lépés: train vs. temp (80/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,  # ← FONTOS: osztályarányok megőrzése!
    )

    # 2. lépés: validation vs. test (50/50 a temp-ből → 10/10)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print(f"  Train:      {X_train.shape[0]} minta ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"  Validation: {X_val.shape[0]} minta ({X_val.shape[0]/len(X)*100:.0f}%)")
    print(f"  Test:       {X_test.shape[0]} minta ({X_test.shape[0]/len(X)*100:.0f}%)")

    # Osztályarányok ellenőrzése
    for nev, y_rész in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {nev} osztályarány: {np.bincount(y_rész) / len(y_rész)}")


# ════════════════════════════════════════════════════════════
# 2. CROSS-VALIDATION
# ════════════════════════════════════════════════════════════

def cross_validation_demo() -> None:
    """K-Fold cross-validation összehasonlítás.

    MIÉRT CROSS-VALIDATION?
      Egyetlen split eredménye véletlenszerű lehet.
      CV k-szorosán értékeli a modellt → megbízhatóbb becslés.

    K megválasztása:
      k=5:  Gyors, jó kompromisszum
      k=10: Stabilabb, de lassabb
      LOOCV (k=n): Kis adatnál, de nagyon lassú
    """
    print("\n=== 2. CROSS-VALIDATION ===")

    X, y = make_classification(n_samples=800, n_features=15, random_state=42, weights=[0.65, 0.35])

    modellek = {
        "Logisztikus Reg.": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for nev, modell in modellek.items():
        # Stratified K-Fold (osztályarányokat megőriz minden fold-ban)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(modell, X, y, cv=skf, scoring="roc_auc")
        print(f"  {nev}: ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"    Fold-ok: {scores.round(4)}")


# ════════════════════════════════════════════════════════════
# 3. IDŐSOR CROSS-VALIDATION
# ════════════════════════════════════════════════════════════

def idosor_cv_demo() -> None:
    """TimeSeriesSplit – idősorhoz KÖTELEZŐ.

    ⚠️  Idősorban NEM lehet random split!
        Jövőbeli adatot nem használhatunk múltbeli előrejelzésnél.
        TimeSeriesSplit: mindig csak a múlt → jelenre tanít.
    """
    print("\n=== 3. IDŐSOR CV ===")

    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 5))
    y = (np.cumsum(rng.normal(0, 1, n)) > 0).astype(int)

    tss = TimeSeriesSplit(n_splits=5)

    print("  TimeSeriesSplit fold-ok:")
    for fold, (train_idx, test_idx) in enumerate(tss.split(X), 1):
        print(f"    Fold {fold}: train [{train_idx[0]}–{train_idx[-1]}], "
              f"test [{test_idx[0]}–{test_idx[-1]}]")

    # Vizualizáció
    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    for fold, (train_idx, test_idx) in enumerate(tss.split(X)):
        axes[fold].scatter(train_idx, [fold+1]*len(train_idx), c="#3498DB", s=5, label="train")
        axes[fold].scatter(test_idx,  [fold+1]*len(test_idx),  c="#E74C3C", s=5, label="test")
        axes[fold].set_ylabel(f"Fold {fold+1}", fontsize=8)
        axes[fold].set_yticks([])

    axes[0].set_title("TimeSeriesSplit – Időrendi sorrend megőrzése", fontsize=12)
    axes[0].legend(loc="upper right", markerscale=3)
    axes[-1].set_xlabel("Időindex")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_timeseries_split.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# 4. LEARNING CURVE – OVERFITTING DIAGNÓZIS
# ════════════════════════════════════════════════════════════

def learning_curve_plot() -> None:
    """Learning curve: overfitting vs. underfitting vizualizálása.

    HA:
      train_score >> val_score → OVERFITTING (túl komplex modell)
      mind kettő alacsony      → UNDERFITTING (túl egyszerű, több adat kell)
      konvergálnak             → JÓ modell
    """
    print("\n=== 4. LEARNING CURVE ===")

    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=50, random_state=42),
        X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#2ECC71", label="Train score")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="#2ECC71")
    plt.plot(train_sizes, val_mean, "o-", color="#E74C3C", label="Validation score")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="#E74C3C")
    plt.xlabel("Tanítóminták száma")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve – Overfitting diagnosztika")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "17_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Diagnózis
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.10:
        print(f"  ⚠️  OVERFITTING detektálva! Train-Val rés: {gap:.3f}")
    elif val_mean[-1] < 0.70:
        print(f"  ⚠️  UNDERFITTING! Validation score csak: {val_mean[-1]:.3f}")
    else:
        print(f"  ✅ Egészséges modell! Val score: {val_mean[-1]:.3f}, rés: {gap:.3f}")


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LECKE 17 – Train/Test Split & Cross-Validation")
    print("=" * 55)

    alap_felosztás_demo()
    cross_validation_demo()
    idosor_cv_demo()
    learning_curve_plot()

    print("\n✅ Lecke 17 sikeresen lefutott!")
    print("➡️  Következő: 16_ml_machine_learn.py")