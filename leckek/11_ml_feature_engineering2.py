"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 11 – Feature Engineering II.                          ║
║  Idősor-jellemzők, Target Encoding, Feature Selection        ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - Dátum/idősor jellemzők kinyerése
  - Rolling window és lag jellemzők
  - Target Encoding (category_encoders)
  - Feature selection: VarianceThreshold, RFECV, SHAP
  - Magas kardinalitású kategorikus változók kezelése
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import (
        RFE,
        SelectKBest,
        VarianceThreshold,
        f_classif,
        mutual_info_classif,
    )
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
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
# IDŐSOR JELLEMZŐK
# ════════════════════════════════════════════════════════════

def idosor_jellemzok(df: pd.DataFrame, datum_col: str = "datum") -> pd.DataFrame:
    """Dátumból jellemzők kinyerése.

    Ciklikus encoding (sin/cos): a hetek, hónapok ciklikusak!
    pl. December (12) és Január (1) közel vannak → sin/cos jobb, mint sima szám.

    Args:
        df:        DataFrame dátum oszloppal.
        datum_col: Dátum oszlop neve.

    Returns:
        DataFrame kiegészített idősor-jellemzőkkel.
    """
    df = df.copy()
    dt = pd.to_datetime(df[datum_col])

    # Alap komponensek
    df["ev"]      = dt.dt.year
    df["honap"]   = dt.dt.month
    df["het_nap"] = dt.dt.dayofweek   # 0=Hétfő, 6=Vasárnap
    df["ev_nap"]  = dt.dt.dayofyear
    df["het"]     = dt.dt.isocalendar().week.astype(int)
    df["negydev"] = dt.dt.quarter

    # Ciklikus encoding – sin/cos (kör-reprezentáció)
    df["honap_sin"] = np.sin(2 * np.pi * df["honap"] / 12)
    df["honap_cos"] = np.cos(2 * np.pi * df["honap"] / 12)
    df["het_nap_sin"] = np.sin(2 * np.pi * df["het_nap"] / 7)
    df["het_nap_cos"] = np.cos(2 * np.pi * df["het_nap"] / 7)

    # Bináris jellemzők
    df["hetetvege"]   = (df["het_nap"] >= 5).astype(int)
    df["ev_eleje"]    = (df["honap"] <= 3).astype(int)
    df["ev_vege"]     = (df["honap"] >= 10).astype(int)

    return df


def rolling_jellemzok(df: pd.DataFrame, col: str, group_col: str | None = None) -> pd.DataFrame:
    """Rolling window (mozgóablak) jellemzők.

    Idősorokban a múltbeli értékek erős előrejelzők lehetnek.
    LAG: az x napos múltbeli érték
    ROLLING MEAN/STD: az elmúlt x nap átlaga/szórása

    ⚠️  FIGYELEM: Mindig időrendi sorrendben számold!
        Különben data leakage! (jövőbeli adatot használsz múltbeli előrejelzéshez)

    Args:
        df:        Idősor DataFrame.
        col:       A célfüggvény neve.
        group_col: Ha van csoport (pl. vállalat), ott külön számol.

    Returns:
        DataFrame lag/rolling jellemzőkkel.
    """
    df = df.copy().sort_values("datum")

    if group_col:
        grp = df.groupby(group_col)[col]
    else:
        grp = df[col]

    # Lag jellemzők
    for lag in [1, 3, 7, 14, 30]:
        df[f"{col}_lag_{lag}"] = grp.shift(lag)

    # Rolling statisztikák
    for ablak in [7, 14, 30]:
        df[f"{col}_roll_mean_{ablak}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=ablak, min_periods=3).mean()
        )
        df[f"{col}_roll_std_{ablak}"] = grp.transform(
            lambda x: x.shift(1).rolling(window=ablak, min_periods=3).std()
        )

    # Momentum: jelenlegi vs. x nap előtti arány
    df[f"{col}_momentum_7"] = df[col] / (df[f"{col}_lag_7"] + 1e-8)

    return df


# ════════════════════════════════════════════════════════════
# TARGET ENCODING
# ════════════════════════════════════════════════════════════

def target_encoding(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cat_col: str,
    target_col: str,
    simitas: float = 10.0,
) -> tuple[pd.Series, pd.Series]:
    """Target (Mean) Encoding overfitting-ellenes simítással.

    Hogyan működik:
      Minden kategóriához a célváltozó átlagát tárolja.
      pl. varos="Budapest" → átlagos nemfizetés 12.3%

    Simítás (smoothing):
      Kis mintaszámú kategóriák az összátlaghoz húznak.
      encoded = (n * cat_mean + k * global_mean) / (n + k)
      ahol k a simítási erősség.

    ⚠️  Csak train adaton fitteljük, test adaton csak transformálunk!
        K-Fold target encoding (LeaveOneOut) még jobb overfitting ellen.

    Args:
        df_train:  Tanító adatok.
        df_test:   Teszt adatok.
        cat_col:   Kategorikus oszlop neve.
        target_col: Célváltozó neve.
        simitas:   Simítási erősség (default 10).

    Returns:
        (train_encoded, test_encoded) Series tuple.
    """
    global_mean = df_train[target_col].mean()

    # Kategóriánkénti statisztikák (train-en!)
    stats = df_train.groupby(cat_col)[target_col].agg(["mean", "count"])
    stats.columns = ["cat_mean", "n"]

    # Simítás: kis elemszámnál global_mean felé húz
    stats["encoded"] = (
        (stats["n"] * stats["cat_mean"] + simitas * global_mean)
        / (stats["n"] + simitas)
    )

    # Apply
    train_encoded = df_train[cat_col].map(stats["encoded"]).fillna(global_mean)
    test_encoded  = df_test[cat_col].map(stats["encoded"]).fillna(global_mean)

    return train_encoded, test_encoded


# ════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ════════════════════════════════════════════════════════════

def feature_selection_demo(X: pd.DataFrame, y: pd.Series) -> dict[str, list[str]]:
    """Jellemzőkiválasztás különböző módszerekkel.

    1. VarianceThreshold: Alacsony varianciájú (majdnem konstans) jellemzők eltávolítása
    2. Statisztikai teszt (F-statisztika, mutual info): Univariáns kiválasztás
    3. RFE (Recursive Feature Elimination): Modell-alapú iteratív eltávolítás
    4. Feature Importance (Random Forest): Fa-alapú fontossági sorrend

    Returns:
        Dict módszernév → kiválasztott jellemzők listája.
    """
    eredmenyek: dict[str, list[str]] = {}

    # 1. VarianceThreshold – majdnem konstans jellemzők ki
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X)
    eredmenyek["variance_threshold"] = X.columns[vt.get_support()].tolist()
    print(f"\n  VarianceThreshold: {len(eredmenyek['variance_threshold'])}/{len(X.columns)} jellemző megmaradt")

    # 2. SelectKBest (F-teszt)
    k = min(10, X.shape[1])
    skb = SelectKBest(f_classif, k=k)
    skb.fit(X.fillna(0), y)
    eredmenyek["f_teszt"] = X.columns[skb.get_support()].tolist()
    print(f"  SelectKBest (F-teszt, k={k}): {eredmenyek['f_teszt']}")

    # 3. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    eredmenyek["rf_importance"] = importances.head(10).index.tolist()

    # Vizualizáció
    plt.figure(figsize=(10, 5))
    importances.head(15).plot(kind="bar", color="#3498DB", alpha=0.8)
    plt.title("Random Forest Feature Importance (Top 15)")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    return eredmenyek


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LECKE 11 – Feature Engineering II.")
    print("=" * 55)

    # Szintetikus adathalmaz dátummal
    rng = np.random.default_rng(42)
    n = 600

    df = pd.DataFrame({
        "datum":    pd.date_range("2022-01-01", periods=n, freq="D"),
        "ertek":    rng.integers(100, 500, n).astype(float),
        "varos":    rng.choice(["Budapest", "Debrecen", "Pécs"] * 5 + ["X_" + str(i) for i in range(50)], n),
        "cel":      rng.integers(0, 2, n),
    })

    # Idősor jellemzők
    print("\n--- Idősor jellemzők ---")
    df = idosor_jellemzok(df, "datum")
    print(f"Új idősor jellemzők: {[c for c in df.columns if c not in ['datum', 'ertek', 'varos', 'cel']]}")

    # Rolling jellemzők
    print("\n--- Rolling jellemzők ---")
    df = rolling_jellemzok(df, "ertek")
    df = df.dropna()  # lag miatti NaN-ok eltávolítása

    # Target encoding
    print("\n--- Target Encoding ---")
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    train_enc, test_enc = target_encoding(df_train, df_test, "varos", "cel")
    df_train["varos_encoded"] = train_enc.values
    df_test["varos_encoded"]  = test_enc.values
    print(f"  Varos → target encoded: {df_train['varos_encoded'].describe().to_dict()}")

    # Feature selection
    print("\n--- Feature Selection ---")
    df_all = pd.concat([df_train, df_test])
    X = df_all.select_dtypes(include=[np.number]).drop("cel", axis=1, errors="ignore")
    y = df_all["cel"]
    kivalasztott = feature_selection_demo(X, y)

    print("\n✅ Lecke 11 sikeresen lefutott!")
    print("➡️  Következő: 17_ml_train_test_split.py")