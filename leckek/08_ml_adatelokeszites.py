"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 08 – ML Adatelőkészítés (Preprocessing Pipeline)      ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - Encoding: OrdinalEncoder, OneHotEncoder, TargetEncoder
  - Scaling: StandardScaler, MinMaxScaler, RobustScaler
  - sklearn Pipeline – a helyes workflow
  - ColumnTransformer – különböző típusok egyszerre
  - Data leakage elkerülése (KRITIKUS!)

ARANY SZABÁLY:
  Fit CSAK a train adaton! Transform-ot alkalmazd train ÉS test adatokra.
  Ha a test adaton is fittelsz → data leakage → hamis teljesítmény!
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        RobustScaler,
        StandardScaler,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
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

def adat_betoltes() -> pd.DataFrame:
    """Szintetikus adathalmaz hitelkockázat előrejelzéshez.

    Tartalmaz:
      - Numerikus, kategorikus és ordinális oszlopokat
      - Hiányzó értékeket (valóság-szimulálás)

    Returns:
        Pandas DataFrame.
    """
    rng = np.random.default_rng(42)
    n = 1000

    kor = rng.integers(20, 65, n).astype(float)
    bevetel = rng.integers(100_000, 900_000, n).astype(float)
    hitel_osszeg = rng.integers(500_000, 5_000_000, n).astype(float)
    foglalkozas = rng.choice(["alkalmazott", "vállalkozó", "diák", "nyugdíjas"], n)
    vegzettseg = rng.choice(["alap", "közép", "felső"], n)  # ordinális!
    varos = rng.choice(["Budapest", "Debrecen", "Pécs", "Győr"], n)
    cel = (bevetel * 0.4 - hitel_osszeg * 0.3 + rng.normal(0, 50000, n) > 0).astype(int)

    df = pd.DataFrame({
        "kor": kor, "bevetel": bevetel, "hitel_osszeg": hitel_osszeg,
        "foglalkozas": foglalkozas, "vegzettseg": vegzettseg,
        "varos": varos, "veszely": cel,
    })

    # Hiányzó értékek injektálása (~8%)
    for col in ["kor", "bevetel", "foglalkozas"]:
        mask = rng.random(n) < 0.08
        df.loc[mask, col] = np.nan

    return df


# ════════════════════════════════════════════════════════════
# 1. SCALING – MIKOR MELYIKET?
# ════════════════════════════════════════════════════════════

def scaling_demo(df: pd.DataFrame) -> None:
    """StandardScaler vs. MinMaxScaler vs. RobustScaler összehasonlítás.

    StandardScaler:  (x - mean) / std  →  mean=0, std=1
                     Érzékeny outlierekre. Legtöbb ML algo-hoz jó.

    MinMaxScaler:    (x - min) / (max - min)  →  [0, 1]
                     Neuronhálókhoz, kNN-hez ajánlott.

    RobustScaler:    (x - medián) / IQR
                     Outlier-robusztus. Ha sok a szélső érték.
    """
    col = "bevetel"
    x = df[col].dropna().values.reshape(-1, 1)

    scalers = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler":   MinMaxScaler(),
        "RobustScaler":   RobustScaler(),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].hist(x, bins=30, color="gray", alpha=0.7)
    axes[0].set_title("Eredeti")

    for ax, (nev, scaler) in zip(axes[1:], scalers.items()):
        x_scaled = scaler.fit_transform(x)
        ax.hist(x_scaled, bins=30, alpha=0.7)
        ax.set_title(nev)
        ax.set_xlabel(f"mean={x_scaled.mean():.2f}, std={x_scaled.std():.2f}")

    plt.suptitle("Scaling módszerek összehasonlítása", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_scaling_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# 2. ENCODING – KATEGORIKUS VÁLTOZÓK
# ════════════════════════════════════════════════════════════

def encoding_demo(df: pd.DataFrame) -> None:
    """Encoding módszerek összehasonlítása.

    OneHotEncoder:   Nominális változókhoz (nincs sorrend).
                     Pl.: varos → [Budapest, Debrecen, Pécs, Győr]
                     Hátrány: sok kategóriánál magas dimenzió (curse of dim.)

    OrdinalEncoder:  Ordinális változókhoz (van sorrend).
                     Pl.: alap < közép < felső

    TargetEncoder:   Magas kardinalitású változókhoz.
                     Lecke 11-ben részletesen.
    """
    print("\n=== ENCODING DEMO ===")

    # One-Hot Encoding
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    varos_encoded = ohe.fit_transform(df[["varos"]].dropna())
    print(f"\nOneHot: {ohe.get_feature_names_out()}")
    print(varos_encoded[:3])

    # Ordinális Encoding – sorrend megadásával!
    ord_enc = OrdinalEncoder(
        categories=[["alap", "közép", "felső"]],   # ← KÖTELEZŐ a sorrend megadása
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    vegz_encoded = ord_enc.fit_transform(df[["vegzettseg"]].dropna())
    print(f"\nOrdinális: alap=0, közép=1, felső=2")
    print(pd.Series(vegz_encoded.ravel()).value_counts().sort_index())


# ════════════════════════════════════════════════════════════
# 3. SKLEARN PIPELINE – AZ ARANY SZABÁLY
# ════════════════════════════════════════════════════════════

def pipeline_epites(df: pd.DataFrame) -> Pipeline:
    """Teljes preprocessing + modell pipeline.

    Miért Pipeline?
      1. Nem lehet data leakage: fit/transform szétválasztva
      2. Könnyen reprodukálható és deployolható
      3. Cross-validation és hyperparaméter-hangolás egyben működik

    Returns:
        Fitteletlen sklearn Pipeline.
    """
    # Oszlopok típus szerint csoportosítva
    num_cols = ["kor", "bevetel", "hitel_osszeg"]
    cat_cols = ["foglalkozas", "varos"]
    ord_cols = ["vegzettseg"]

    # Numerikus ág: imputálás → scaling
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    # Kategorikus ág: imputálás → one-hot
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Ordinális ág: imputálás → ordinális encoding
    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["alap", "közép", "felső"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    # ColumnTransformer – egyszerre kezeli az összes ágat
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
        ("ord", ord_pipeline, ord_cols),
    ], remainder="drop")  # többi oszlopot eldobja

    # Teljes pipeline: preprocessing + modell
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    return pipeline


def pipeline_kiartekelese(df: pd.DataFrame) -> None:
    """Pipeline kiértékelése cross-validation-nel.

    Args:
        df: Az adathalmaz.
    """
    print("\n=== PIPELINE KIÉRTÉKELÉS ===")

    X = df.drop("veszely", axis=1)
    y = df["veszely"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = pipeline_epites(df)

    # Cross-validation – CSAK a train adaton!
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"CV accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Végső kiértékelés
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nKlasszifikációs riport (test set):")
    print(classification_report(y_test, y_pred, target_names=["Nem veszélyes", "Veszélyes"]))

    return pipeline


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LECKE 08 – ML Adatelőkészítés")
    print("=" * 55)

    df = adat_betoltes()
    print(f"Adat shape: {df.shape}")
    print(df.head(3).to_string())

    scaling_demo(df)
    encoding_demo(df)
    pipeline_kiartekelese(df)

    print("\n✅ Lecke 08 sikeresen lefutott!")
    print("➡️  Következő: 10_ml_feature_engineering1.py")