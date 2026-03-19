"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 14 – Lineáris Regresszió (Ridge, Lasso, ElasticNet)  ║
╚══════════════════════════════════════════════════════════════╝
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import warnings; warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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
# ADATHALMAZ
# ════════════════════════════════════════════════════════════

def regresszio_adat() -> tuple:
    """Ház árazás szintetikus adathalmaz."""
    rng = np.random.default_rng(42)
    n = 600

    alapter = rng.integers(30, 200, n).astype(float)
    szobak  = rng.integers(1, 8, n).astype(float)
    emelet  = rng.integers(0, 20, n).astype(float)
    korzetbe = rng.choice([1, 2, 3, 4, 5], n).astype(float)

    # Valós összefüggés + zaj
    ar = (alapter * 450_000 + szobak * 3_000_000 + emelet * 200_000
          - korzetbe * 500_000 + rng.normal(0, 2_000_000, n))

    X = pd.DataFrame({"alapter": alapter, "szobak": szobak, "emelet": emelet, "körzet": korzetbe})
    y = pd.Series(ar, name="ar_huf")
    return X, y


def metrikak(y_true, y_pred, nev: str = "") -> dict:
    """Regressziós metrikák kiírása és visszaadása."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{nev}] RMSE: {rmse:>12,.0f} Ft  |  MAE: {mae:>12,.0f} Ft  |  R²: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ════════════════════════════════════════════════════════════
# REGULÁLT REGRESSZIÓ
# ════════════════════════════════════════════════════════════

def regulalt_regresszio_demo() -> None:
    """Ridge, Lasso és ElasticNet összehasonlítás.

    OLS (Ordinary Least Squares):
      Minimalizálja: Σ(y - ŷ)²
      Gond: Magas dimenziónál, multikollinearitásnál overfittingel.

    Ridge (L2 regularizáció):
      + λ·Σβ² → Koefficiensek kicsik maradnak, de NEM nullázódnak.
      Jó: Minden jellemző fontos, csak csökkenteni kell.

    Lasso (L1 regularizáció):
      + λ·Σ|β| → Koefficienseket NULLÁRA húzza → automatikus feature selection!
      Jó: Feltételezed, hogy csak néhány jellemző fontos.

    ElasticNet:
      Ridge + Lasso keveréke → rugalmas, sok jellemzőnél ajánlott.
    """
    print("\n=== LECKE 14 – Regulált Regresszió ===")

    X, y = regresszio_adat()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modellek = {
        "OLS":        LinearRegression(),
        "Ridge":      Ridge(alpha=1e5),
        "Lasso":      Lasso(alpha=1e4, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=1e4, l1_ratio=0.5, max_iter=10000),
    }

    for nev, modell in modellek.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", modell)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrikak(y_test, y_pred, nev)

    # RidgeCV – automatikusan keresi az optimális alpha-t
    ridgecv = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=[1e3, 1e4, 1e5, 1e6], cv=5)),
    ])
    ridgecv.fit(X_train, y_train)
    print(f"\n  Optimális Ridge alpha: {ridgecv.named_steps['model'].alpha_:.0f}")
    metrikak(y_test, ridgecv.predict(X_test), "RidgeCV (auto alpha)")

    # Maradékok vizualizálása
    y_pred_final = ridgecv.predict(X_test)
    maradek = y_test - y_pred_final

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred_final / 1e6, maradek / 1e6, alpha=0.4, color="#3498DB")
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("Előrejelzett érték (MFt)")
    axes[0].set_ylabel("Maradék (MFt)")
    axes[0].set_title("Maradékok eloszlása")

    axes[1].scatter(y_test / 1e6, y_pred_final / 1e6, alpha=0.4, color="#2ECC71")
    mn, mx = y_test.min() / 1e6, y_test.max() / 1e6
    axes[1].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[1].set_xlabel("Valós érték (MFt)")
    axes[1].set_ylabel("Előrejelzett (MFt)")
    axes[1].set_title("Valós vs. Előrejelzett")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "14_regresszio_maradek.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# LECKE 15 – LOGISZTIKUS REGRESSZIÓ
# ════════════════════════════════════════════════════════════

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV


def logisztikus_regresszio_demo() -> None:
    """Logisztikus Regresszió: bináris és multi-class.

    Sigmoid függvény: σ(z) = 1 / (1 + e^(-z))
    Az output valószínűség! Nem osztály.

    Küszöbérték (default: 0.5):
      Lehet módosítani: precision-recall trade-off alapján.
      Orvosi diagnózisnál: alacsonyabb küszöb → több pozitív detektálás.

    C paraméter = 1/λ (regularizáció erőssége)
      Kis C:  erős regularizáció (underfitting veszély)
      Nagy C: gyenge regularizáció (overfitting veszély)
    """
    print("\n\n=== LECKE 15 – Logisztikus Regresszió ===")

    X, y = make_classification(
        n_samples=800, n_features=15, n_informative=8,
        n_classes=2, random_state=42, weights=[0.7, 0.3],  # kiegyensúlyozatlan!
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Modell kiegyensúlyozatlan osztályokra
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",   # ← kiegyensúlyozatlan adatnál fontos!
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Negatív", "Pozitív"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Vizualizáció: Confusion Matrix + ROC
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[0],
                                             display_labels=["Negatív", "Pozitív"],
                                             colorbar=False)
    axes[0].set_title("Konfúziós mátrix")

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_title("ROC görbe")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "15_logisztikus_roc.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Küszöbérték módosítása (pl. recall maximalizálása)
    print("\n  Küszöbérték érzékenység:")
    for kuszob in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_k = (y_proba >= kuszob).astype(int)
        from sklearn.metrics import recall_score, precision_score
        prec = precision_score(y_test, y_pred_k, zero_division=0)
        rec  = recall_score(y_test, y_pred_k, zero_division=0)
        print(f"    Küszöb={kuszob:.1f}: Precision={prec:.3f}, Recall={rec:.3f}")


# ════════════════════════════════════════════════════════════
# LECKE 16 – ML WORKFLOW ÁTTEKINTÉS
# ════════════════════════════════════════════════════════════

def ml_workflow_demo() -> None:
    """Teljes ML workflow összefoglalása.

    LÉPÉSEK:
    1. Probléma definiálása (regresszió / osztályozás / klaszterezés)
    2. Adatgyűjtés és explorálás (EDA)
    3. Adattisztítás (lecke 02)
    4. Feature engineering (lecke 10-11)
    5. Adatfelosztás (lecke 17)
    6. Preprocessing pipeline (lecke 08)
    7. Modell kiválasztása és tanítása
    8. Kiértékelés (megfelelő metrika!)
    9. Hyperparaméter-hangolás (lecke 09)
    10. Modell mentése és deployolása (lecke 19)
    """
    print("\n\n=== LECKE 16 – ML Workflow Összefoglalás ===")

    metrikak_tablazat = pd.DataFrame({
        "Feladat":    ["Regresszió",   "Osztályozás", "Osztályozás",   "Osztályozás"],
        "Metrika":    ["RMSE, MAE, R²", "Accuracy",   "ROC-AUC",       "F1, Precision, Recall"],
        "Mikor?":     ["Mindig",        "Kiegyensúlyozott adat", "Kiegyensúlyozatlan", "Ha egyik sem jó önmagában"],
        "Sklearn":    ["mean_squared_error", "accuracy_score", "roc_auc_score", "f1_score"],
    })

    print("\nMetrika útmutató:")
    print(metrikak_tablazat.to_string(index=False))

    modell_tabla = pd.DataFrame({
        "Modell":             ["Lin. Reg.", "Log. Reg.", "Random Forest", "XGBoost", "SVM", "kNN"],
        "Sebesség":           ["⚡⚡⚡",   "⚡⚡⚡",    "⚡⚡",          "⚡⚡",     "⚡",    "⚡"],
        "Pontosság":          ["közepes",   "közepes",   "magas",         "magas",    "magas", "közepes"],
        "Magyarázhatóság":    ["✅ igen",    "✅ igen",   "⚠️ részben",    "⚠️ részben","❌ nem","❌ nem"],
        "Nagy adatnál":       ["✅",         "✅",        "✅",             "✅",        "❌",    "❌"],
    })

    print("\nModell összehasonlítás:")
    print(modell_tabla.to_string(index=False))


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    regulalt_regresszio_demo()
    logisztikus_regresszio_demo()
    ml_workflow_demo()

    print("\n✅ Leckék 14/15/16 sikeresen lefutottak!")
    print("➡️  Következő: 07_ml_random_forest.py")