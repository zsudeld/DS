"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 09 – AutoML (Optuna + FLAML)                          ║
║  LECKE 19 – Experiment Tracking (MLflow)                     ║
╚══════════════════════════════════════════════════════════════╝
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import warnings; warnings.filterwarnings("ignore")
    import os

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
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
# LECKE 09 – AUTOML
# ════════════════════════════════════════════════════════════

def optuna_demo() -> None:
    """Optuna – Bayesiánus hyperparaméter-optimalizálás.

    MIÉRT OPTUNA A GRIDSEARCH HELYETT?
      GridSearch: összes kombináció kipróbálása → exponenciális
      RandomSearch: véletlenszerű → nem hatékony
      Optuna (TPE): Bayes-i megközelítés → intelligensen keres

    TRIAL-PRUNING: Gyengén teljesítő kísérleteket korán abbahagyja!
    """
    print("=" * 55)
    print("  LECKE 09 – AutoML (Optuna)")
    print("=" * 55)

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Optuna nem telepítve: pip install optuna")
        return

    X, y = make_classification(n_samples=1000, n_features=15, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective függvény – ezt maximalizáljuk.

        Args:
            trial: Optuna trial objektum.

        Returns:
            Cross-validation ROC-AUC átlag.
        """
        # Modellválasztás (kategorikus paraméter)
        modell_nev = trial.suggest_categorical("model", ["rf", "lr"])

        if modell_nev == "rf":
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 50, 300),
                "max_depth":       trial.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":    trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
            modell = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        else:
            C = trial.suggest_float("C", 1e-3, 1e2, log=True)  # log-skála!
            modell = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=1000, random_state=42)),
            ])

        score = cross_val_score(modell, X_train, y_train, cv=3, scoring="roc_auc").mean()
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"\n  Legjobb ROC-AUC (CV): {study.best_value:.4f}")
    print(f"  Legjobb paraméterek:   {study.best_params}")

    # Vizualizáció: optimalizáció fejlődése
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title("Optuna – Optimalizáció fejlődése")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "09_optuna_history.png", dpi=150, bbox_inches="tight")
        plt.show()
    except Exception:
        # Egyszerű alternatív plot
        values = [t.value for t in study.trials if t.value is not None]
        best_so_far = [max(values[:i+1]) for i in range(len(values))]
        plt.figure(figsize=(8, 4))
        plt.plot(values, ".", alpha=0.4, label="Trial score")
        plt.plot(best_so_far, "-", color="red", label="Legjobb eddig")
        plt.xlabel("Trial")
        plt.ylabel("ROC-AUC")
        plt.title("Optuna – Optimalizáció fejlődése")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "09_optuna_history.png", dpi=150, bbox_inches="tight")
        plt.show()


def flaml_demo() -> None:
    """FLAML – Gyors, könnyűsúlyú AutoML.

    FLAML előnyei:
      - Nagyon alacsony memóriaigény
      - Időlimit alapú futtatás
      - Beépített pipeline (preprocessing + modell)
      - Üzleti felhasználásra optimalizált
    """
    print("\n--- FLAML Demo ---")
    try:
        from flaml import AutoML
    except ImportError:
        print("  FLAML nem telepítve: pip install flaml")
        return

    X, y = make_classification(n_samples=800, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    automl = AutoML()
    automl.fit(
        X_train, y_train,
        task="classification",
        metric="roc_auc",
        time_budget=30,       # Maximum 30 másodperc
        verbose=0,
    )

    y_pred_proba = automl.predict_proba(X_test)[:, 1]
    print(f"  FLAML legjobb modell: {automl.best_estimator}")
    print(f"  FLAML ROC-AUC (test): {roc_auc_score(y_test, y_pred_proba):.4f}")


# ════════════════════════════════════════════════════════════
# LECKE 19 – MLFLOW EXPERIMENT TRACKING
# ════════════════════════════════════════════════════════════

def mlflow_demo() -> None:
    """MLflow – Modell kísérletek nyomon követése.

    MIÉRT KELL TRACKING?
      Nélküle: "Melyik modell volt a legjobb? Milyen paraméterekkel?"
      MLflow-val: Minden futás rögzítve, összehasonlítható!

    NÉGY FŐ KOMPONENS:
      1. Tracking: metrikák, paraméterek, artefaktok logolása
      2. Models: modell formátumok (sklearn, pytorch, etc.)
      3. Projects: reprodukálható projektek
      4. Registry: modell verziókezelés és deployment

    INDÍTÁS: mlflow ui  (böngészőben: http://localhost:5000)
    """
    print("\n" + "=" * 55)
    print("  LECKE 19 – MLflow Experiment Tracking")
    print("=" * 55)

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("  MLflow nem telepítve: pip install mlflow")
        _mlflow_nelkul_demo()
        return

    X, y = make_classification(n_samples=800, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    os.makedirs("models", exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")   # helyi tárolás
    mlflow.set_experiment("ds_kurzus_demo")

    # Kísérlet-paraméterek (különböző konfigurációk)
    kiserlet_konfiguracik = [
        {"n_estimators": 100, "max_depth": 5,  "nev": "RF_small"},
        {"n_estimators": 200, "max_depth": 10, "nev": "RF_medium"},
        {"n_estimators": 300, "max_depth": 15, "nev": "RF_large"},
    ]

    for konfig in kiserlet_konfiguracik:
        with mlflow.start_run(run_name=konfig["nev"]):
            # Paraméterek logolása
            mlflow.log_param("n_estimators", konfig["n_estimators"])
            mlflow.log_param("max_depth",    konfig["max_depth"])
            mlflow.log_param("model_type",   "RandomForest")

            # Tanítás
            modell = RandomForestClassifier(
                n_estimators=konfig["n_estimators"],
                max_depth=konfig["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
            modell.fit(X_train, y_train)

            # Metrikák logolása
            y_proba = modell.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            cv_score = cross_val_score(modell, X_train, y_train, cv=3, scoring="roc_auc").mean()

            mlflow.log_metric("roc_auc_test", auc)
            mlflow.log_metric("roc_auc_cv",   cv_score)

            # Modell mentése
            mlflow.sklearn.log_model(modell, "modell")

            print(f"  [{konfig['nev']}] ROC-AUC: {auc:.4f}  (CV: {cv_score:.4f})")

    print("\n  ✅ MLflow kísérletek rögzítve!")
    print("  📊 Megjelenítés: mlflow ui")
    print("  🌐 Böngésző: http://localhost:5000")


def _mlflow_nelkul_demo() -> None:
    """Egyszerű kézi logolás MLflow nélkül – JSON-ba."""
    import json
    from datetime import datetime

    X, y = make_classification(n_samples=800, n_features=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    futasok = []
    for n_est in [50, 100, 200]:
        modell = RandomForestClassifier(n_estimators=n_est, random_state=42)
        modell.fit(X_train, y_train)
        auc = roc_auc_score(y_test, modell.predict_proba(X_test)[:, 1])
        futasok.append({
            "timestamp":    datetime.now().isoformat(),
            "n_estimators": n_est,
            "roc_auc":      round(auc, 4),
        })
        print(f"  [n_estimators={n_est}] ROC-AUC: {auc:.4f}")

    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_DIR / "19_experiment_log.json", "w", encoding="utf-8") as f:
        json.dump(futasok, f, indent=2)
    print("  📁 Kísérletek mentve: outputs/19_experiment_log.json")


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    optuna_demo()
    flaml_demo()
    mlflow_demo()

    print("\n✅ Leckék 09/19 sikeresen lefutottak!")
    print("➡️  Következő: 18_plotly_express.py")