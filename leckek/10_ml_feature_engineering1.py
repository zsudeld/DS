"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 10 – Feature Engineering I.                           ║
║  Új jellemzők létrehozása, binning, interakciók              ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - Domain-specifikus jellemzők kézzel
  - Binning (pd.cut, pd.qcut)
  - Matematikai transzformációk (log, sqrt, power)
  - Interakció-tagok és polinomiális jellemzők
  - Arány-jellemzők

MIÉRT FONTOS?
  Egy jó jellemző fontosabb lehet, mint a legjobb algoritmus!
  "Feature engineering is the art of the craft." – Andrew Ng
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:

    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
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
    rng = np.random.default_rng(42)
    n = 800

    kor = rng.integers(20, 65, n).astype(float)
    bevetel = rng.integers(150_000, 1_200_000, n).astype(float)
    hitel = rng.integers(500_000, 6_000_000, n).astype(float)
    fut_ido_ho = rng.integers(12, 120, n)  # hónapok
    tartozas = rng.integers(0, 500_000, n).astype(float)

    cel = ((bevetel / hitel) * fut_ido_ho - tartozas * 0.5 + rng.normal(0, 50, n) > 20).astype(int)

    return pd.DataFrame({
        "kor": kor, "bevetel": bevetel, "hitel": hitel,
        "fut_ido_ho": fut_ido_ho, "tartozas": tartozas,
        "cel": cel,
    })


# ════════════════════════════════════════════════════════════
# 1. DOMAIN-SPECIFIKUS JELLEMZŐK
# ════════════════════════════════════════════════════════════

def domain_jellemzok(df: pd.DataFrame) -> pd.DataFrame:
    """Üzleti logikán alapuló jellemzők.

    Ezek a legértékesebbek, mert tartalmazzák a domain-tudást.
    A modell önmagától nem találná meg ezeket.
    """
    df = df.copy()

    # Törlesztési arány: mekkora a havi törlesztő a bevételhez képest
    # (Bankok általában 30-40% felett kockázatosnak tartják)
    df["torleszt_arany"] = (df["hitel"] / df["fut_ido_ho"]) / df["bevetel"]

    # DTI – Debt-to-Income ratio
    df["dti"] = df["tartozas"] / df["bevetel"]

    # Hitel/bevétel szorzó (hányszoros éves bevétel a hitel?)
    df["hitel_bevetel_szorzó"] = df["hitel"] / (df["bevetel"] * 12)

    # Szabad kapacitás: mi marad a törlesztő után
    df["szabad_bevetel"] = df["bevetel"] - df["hitel"] / df["fut_ido_ho"]

    print(f"Új jellemzők hozzáadva. Shape: {df.shape}")
    return df


# ════════════════════════════════════════════════════════════
# 2. BINNING – FOLYTONOS → KATEGORIKUS
# ════════════════════════════════════════════════════════════

def binning_demo(df: pd.DataFrame) -> pd.DataFrame:
    """Binning: folytonos változó diskretizálása.

    pd.cut:  Egyenlő szélességű kategóriák (fix határok)
    pd.qcut: Egyenlő elemszámú kategóriák (kvantilisek)

    Mikor hasznos?
      - Nemlineáris összefüggéseknél
      - Döntési fa + lineáris modell kombinálásakor
      - Domain-alapú küszöbértékeknél
    """
    df = df.copy()

    # Kor kategóriák (domain tudás alapján)
    df["kor_kategoria"] = pd.cut(
        df["kor"],
        bins=[0, 25, 35, 50, 100],
        labels=["fiatal", "középkorú", "érett", "idős"],
        right=True,
    )

    # Bevétel kvantilis alapján (percentilis-kategóriák)
    df["bevetel_kvintilis"] = pd.qcut(
        df["bevetel"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    # Futamidő kategóriák
    df["futamido_kat"] = pd.cut(
        df["fut_ido_ho"],
        bins=[0, 24, 60, 120],
        labels=["rövid", "közepes", "hosszú"],
    )

    return df


# ════════════════════════════════════════════════════════════
# 3. MATEMATIKAI TRANSZFORMÁCIÓK
# ════════════════════════════════════════════════════════════

def matematikai_transzformacio(df: pd.DataFrame) -> pd.DataFrame:
    """Log, sqrt, power transzformációk ferdeség csökkentésére.

    Jobbra ferde eloszlásnál (tipikus bevétel, árak):
      log(x+1) → közelebb normálishoz
      sqrt(x)  → enyhébb, ha vannak nullák

    Box-Cox: adatvezérelt optimális transzformáció (pozitív értékekre)
    """
    df = df.copy()

    # Log transzformáció (jobbra ferde → normálishoz közelít)
    df["log_bevetel"]  = np.log1p(df["bevetel"])   # log(x+1) biztonságos, x=0 esetén sem hibázik
    df["log_hitel"]    = np.log1p(df["hitel"])
    df["log_tartozas"] = np.log1p(df["tartozas"])

    # Sqrt – ha a distribúció kevésbé ferde
    df["sqrt_kor"] = np.sqrt(df["kor"])

    # Vizualizáció: eredeti vs. log
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["bevetel"], bins=40, color="#E74C3C", alpha=0.7)
    axes[0].set_title("Bevétel – eredeti (jobbra ferde)")
    axes[1].hist(df["log_bevetel"], bins=40, color="#2ECC71", alpha=0.7)
    axes[1].set_title("log(Bevétel + 1) – normálisabb")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_log_transform.png", dpi=150, bbox_inches="tight")
    plt.show()

    return df


# ════════════════════════════════════════════════════════════
# 4. INTERAKCIÓ-TAGOK ÉS POLINOMIÁLIS JELLEMZŐK
# ════════════════════════════════════════════════════════════

def interakcio_jellemzok(df: pd.DataFrame) -> pd.DataFrame:
    """Polinomiális és interakciós jellemzők.

    Lineáris modelleknek szüksége van ezekre a nemlineáris
    összefüggések befogásához.

    ⚠️  Vigyázz! degree=3+ esetén jellemzők száma robban:
       n=5 feature, degree=2 → 21 feature
       n=5 feature, degree=3 → 56 feature
    """
    df = df.copy()

    # Kézi interakció (értelmesebb, mint automatikus)
    df["kor_x_dti"]    = df["kor"] * df.get("dti", df["tartozas"] / df["bevetel"])
    df["bevetel_x_ido"] = df["bevetel"] * df["fut_ido_ho"]

    # Sklearn PolynomialFeatures (automatikus, de magyarázhatatlanabb)
    num_cols = ["kor", "bevetel", "hitel"]
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_arr = poly.fit_transform(df[num_cols].fillna(df[num_cols].median()))
    poly_df = pd.DataFrame(
        poly_arr,
        columns=poly.get_feature_names_out(num_cols),
        index=df.index,
    )
    print(f"\nPolinomiális jellemzők (degree=2, interaction_only):")
    print(poly_df.columns.tolist())

    return df


# ════════════════════════════════════════════════════════════
# 5. JELLEMZŐK HATÁSÁNAK MÉRÉSE
# ════════════════════════════════════════════════════════════

def jellemzo_hatas(df: pd.DataFrame, verzio: str = "alap") -> float:
    """Cross-validation score mérése adott jellemzőkészlettel.

    Args:
        df:     DataFrame a jellemzőkkel és 'cel' oszloppal.
        verzio: Leíró név a kiíráshoz.

    Returns:
        Átlagos CV accuracy.
    """
    X = df.drop("cel", axis=1).select_dtypes(include=[np.number]).fillna(0)
    y = df["cel"]

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    mean_score = scores.mean()
    print(f"  [{verzio}] CV accuracy: {mean_score:.4f} ± {scores.std():.4f}  ({X.shape[1]} jellemző)")
    return mean_score


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LECKE 10 – Feature Engineering I.")
    print("=" * 55)

    df = adat_betoltes()
    print(f"\nAlap adat: {df.shape}")

    # Alap teljesítmény mérése
    print("\n📊 Teljesítmény-összehasonlítás:")
    score_alap = jellemzo_hatas(df, "alap")

    # Domain jellemzők
    df = domain_jellemzok(df)
    score_domain = jellemzo_hatas(df, "domain jellemzők")

    # Log transzformáció
    df = matematikai_transzformacio(df)
    score_log = jellemzo_hatas(df, "+ log transform")

    # Binning
    df = binning_demo(df)
    df = interakcio_jellemzok(df)
    score_teljes = jellemzo_hatas(df, "+ binning + interakció")

    print(f"\n📈 Javulás: {score_alap:.4f} → {score_teljes:.4f} "
          f"({(score_teljes - score_alap) * 100:+.2f}%)")

    print("\n✅ Lecke 10 sikeresen lefutott!")
    print("➡️  Következő: 11_ml_feature_engineering2.py")