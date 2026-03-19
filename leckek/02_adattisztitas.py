"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 02 – Adattisztítás (Data Cleaning)                    ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - Hiányzó értékek (NaN) feltárása és kezelése
  - Outlier detektálás és kezelés (IQR, Z-score)
  - Duplikátumok kezelése
  - Adattípus-konverziók és formátumhibák javítása
  - Konzisztencia-ellenőrzések
  - Pandas 2.x újdonságok (copy-on-write, dtype_backend)

VALÓS HELYZET: A DS munkaidő ~60-80%-a adattisztítás!
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
    from scipy import stats
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
# SZINTETIKUS "PISZKOS" ADATHALMAZ GENERÁLÁSA
# (Valós projektben: pd.read_csv("adat.csv"))
# ════════════════════════════════════════════════════════════

def piszkos_adat_generalas(n: int = 500) -> pd.DataFrame:
    """Szándékosan hibás adathalmazt generál oktatási célra.

    A visszaadott DataFrame tartalmaz:
      - Hiányzó értékeket (NaN) véletlenszerűen
      - Outliereket (extrém értékek)
      - Duplikált sorokat
      - Helytelen adattípusokat (szám stringként tárolva)
      - Elírt kategória-értékeket

    Args:
        n: Sorok száma.

    Returns:
        Piszkos DataFrame oktatási célra.
    """
    rng = np.random.default_rng(seed=42)

    df = pd.DataFrame({
        "ugyfel_id":  range(1, n + 1),
        "kor":        rng.integers(18, 70, n).astype(float),
        "bevetel":    rng.integers(150_000, 800_000, n).astype(float),
        "pontszam":   rng.uniform(0, 100, n),
        "varos":      rng.choice(["Budapest", "Debrecen", "Pécs", "Győr", "Miskolc"], n),
        "kategoria":  rng.choice(["A", "B", "C"], n),
        "regisztralt": pd.date_range("2020-01-01", periods=n, freq="h"),
    })

    # ── Hibák injektálása ────────────────────────────────
    # 1. Hiányzó értékek (~10%)
    for col in ["kor", "bevetel", "pontszam"]:
        mask = rng.random(n) < 0.10
        df.loc[mask, col] = np.nan

    # 2. Outlierek
    df.loc[rng.choice(n, 10, replace=False), "bevetel"] = rng.integers(5_000_000, 10_000_000, 10)
    df.loc[rng.choice(n, 5, replace=False), "kor"] = rng.integers(120, 200, 5)

    # 3. Duplikátumok
    duplikat_sorok = df.sample(20, random_state=42)
    df = pd.concat([df, duplikat_sorok], ignore_index=True)

    # 4. Helytelen típus (bevetel stringként)
    df["bevetel_str"] = df["bevetel"].apply(
        lambda x: f"{x:.0f} Ft" if not pd.isna(x) else np.nan
    )

    # 5. Elírt kategóriák
    hibas_idx = rng.choice(len(df), 30, replace=False)
    df.loc[hibas_idx, "varos"] = rng.choice(
        ["budapest", "BUDAPEST", "Budepest", "Deb recen"], 30
    )

    return df


# ════════════════════════════════════════════════════════════
# 1. DIAGNÓZIS – Először mindig nézzük meg az adatot!
# ════════════════════════════════════════════════════════════

def diagnosztika(df: pd.DataFrame) -> None:
    """Átfogó adatminőség-diagnózis.

    Kiírja:
      - Alap statisztikák
      - Hiányzó értékek arányát
      - Duplikátumok számát
      - Adattípusokat
    """
    print("=" * 55)
    print("📋 ADATDIAGNÓZIS")
    print("=" * 55)

    print(f"\nShape: {df.shape[0]} sor × {df.shape[1]} oszlop")
    print(f"\nAdattípusok:\n{df.dtypes.to_string()}")

    # Hiányzó értékek összefoglalója
    hianyzok = df.isnull().sum()
    hianyzok_pct = (hianyzok / len(df) * 100).round(2)
    hianyzok_df = pd.DataFrame({
        "hiányzó_db": hianyzok,
        "hiányzó_%": hianyzok_pct,
    }).query("hiányzó_db > 0").sort_values("hiányzó_%", ascending=False)

    if not hianyzok_df.empty:
        print(f"\n⚠️  Hiányzó értékek:\n{hianyzok_df.to_string()}")
    else:
        print("\n✅ Nincs hiányzó érték!")

    # Duplikátumok
    dup_count = df.duplicated().sum()
    print(f"\n🔁 Duplikált sorok: {dup_count}")

    # Numerikus összefoglaló
    print(f"\n📊 Numerikus leíró statisztika:")
    print(df.describe(include=[np.number]).round(2).to_string())


# ════════════════════════════════════════════════════════════
# 2. HIÁNYZÓ ÉRTÉKEK KEZELÉSE
# ════════════════════════════════════════════════════════════

def hianyzok_kezelese(df: pd.DataFrame) -> pd.DataFrame:
    """Hiányzó értékek kezelési stratégiái.

    Stratégiák:
      - Numerikus: medián imputáció (robusztus, nem érzékeny outlierekre)
      - Kategorikus: módusz (leggyakoribb érték)
      - Idősor: forward fill (megelőző érték átvitele)

    ⚠️  FONTOS: Mindig TRAIN adaton fit, TEST adaton csak transform!
        (Lecke 08-ban részletesen.)

    Args:
        df: Tisztítandó DataFrame.

    Returns:
        Imputált DataFrame (copy).
    """
    df = df.copy()

    print("\n" + "=" * 55)
    print("🔧 HIÁNYZÓ ÉRTÉKEK KEZELÉSE")
    print("=" * 55)

    # Numerikus oszlopok: medián imputáció
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        n_hianyzik = df[col].isna().sum()
        if n_hianyzik > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  {col}: {n_hianyzik} hiányzó → medián ({median_val:.2f}) behelyettesítve")

    # Kategorikus: módusz
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        n_hianyzik = df[col].isna().sum()
        if n_hianyzik > 0:
            modus = df[col].mode()[0]
            df[col] = df[col].fillna(modus)
            print(f"  {col}: {n_hianyzik} hiányzó → módusz ('{modus}') behelyettesítve")

    return df


# ════════════════════════════════════════════════════════════
# 3. OUTLIER DETEKTÁLÁS ÉS KEZELÉS
# ════════════════════════════════════════════════════════════

def outlier_kezeles(
    df: pd.DataFrame,
    col: str,
    modszer: str = "iqr",
    hatarszorzo: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Outlier detektálás IQR vagy Z-score alapján.

    IQR módszer:
      - Q1 - 1.5*IQR alatt  →  outlier
      - Q3 + 1.5*IQR felett →  outlier
      - Robusztus: nem érzékeny a szélső értékekre

    Z-score módszer:
      - |z| > 3 esetén outlier
      - Normál eloszlást feltételez

    Args:
        df:           A DataFrame.
        col:          Az oszlop neve.
        modszer:      "iqr" vagy "zscore".
        hatarszorzo:  IQR-szorzó (default 1.5, szigorúbb: 3.0).

    Returns:
        (Tisztított DataFrame, statisztika dict) tuple.
    """
    df = df.copy()
    eredeti_n = len(df)

    if modszer == "iqr":
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        also_hatar = Q1 - hatarszorzo * IQR
        felso_hatar = Q3 + hatarszorzo * IQR
        outlier_mask = (df[col] < also_hatar) | (df[col] > felso_hatar)

    elif modszer == "zscore":
        z = np.abs(stats.zscore(df[col].dropna()))
        # z-score számítása az egész oszlopra (NaN → False)
        z_full = pd.Series(np.nan, index=df.index)
        z_full[df[col].notna()] = np.abs(stats.zscore(df[col].dropna()))
        outlier_mask = z_full > 3
        also_hatar = df[col].mean() - 3 * df[col].std()
        felso_hatar = df[col].mean() + 3 * df[col].std()

    else:
        raise ValueError(f"Ismeretlen módszer: {modszer}. Válassz: 'iqr' vagy 'zscore'")

    n_outlier = outlier_mask.sum()

    # Opció 1: Eldobás (ha outlier < 5%)
    # df_tiszta = df[~outlier_mask]

    # Opció 2: Capping / Winsorizing (megőrzi a sort, de korlátoz)
    df[col] = df[col].clip(lower=also_hatar, upper=felso_hatar)

    statisztika = {
        "col": col,
        "modszer": modszer,
        "n_outlier": int(n_outlier),
        "arany_%": round(n_outlier / eredeti_n * 100, 2),
        "also_hatar": round(also_hatar, 2),
        "felso_hatar": round(felso_hatar, 2),
    }

    print(f"\n  {col}: {n_outlier} outlier ({statisztika['arany_%']}%) → "
          f"capping [{also_hatar:.0f}, {felso_hatar:.0f}]")

    return df, statisztika


# ════════════════════════════════════════════════════════════
# 4. DUPLIKÁTUMOK ÉS TÍPUSHIBÁK
# ════════════════════════════════════════════════════════════

def duplikat_es_tipushiba_kezelese(df: pd.DataFrame) -> pd.DataFrame:
    """Duplikátumok eltávolítása és adattípus-javítás.

    Args:
        df: Tisztítandó DataFrame.

    Returns:
        Javított DataFrame.
    """
    df = df.copy()

    print("\n" + "=" * 55)
    print("🔧 DUPLIKÁTUMOK & TÍPUSHIBÁK")
    print("=" * 55)

    # Duplikátumok
    n_elotte = len(df)
    df = df.drop_duplicates()
    n_utana = len(df)
    print(f"  Duplikátumok eltávolítva: {n_elotte - n_utana} sor")

    # String → numerikus konverzió
    if "bevetel_str" in df.columns:
        df["bevetel_str_clean"] = (
            df["bevetel_str"]
            .str.replace(" Ft", "", regex=False)
            .str.strip()
            .pipe(pd.to_numeric, errors="coerce")   # hibás → NaN
        )
        print(f"  bevetel_str → numerikus: kész")

    # Kategória-harmonizálás (elírt városnevek)
    def harmonizal_varos(varos: str) -> str:
        """Városnév-harmonizálás: kis-nagybetű, whitespace, elírás."""
        if pd.isna(varos):
            return varos
        ervenyes = {"Budapest", "Debrecen", "Pécs", "Győr", "Miskolc"}
        cleaned = varos.strip().title()   # 'BUDAPEST' → 'Budapest'
        # Fuzzy matching helyett explicit mapping oktatási célra
        javitasok = {
            "Budepest": "Budapest",
            "Deb Recen": "Debrecen",
        }
        return javitasok.get(cleaned, cleaned if cleaned in ervenyes else "Egyéb")

    df["varos"] = df["varos"].apply(harmonizal_varos)
    df["varos"] = df["varos"].astype("category")   # memória-hatékony tárolás
    print(f"  Városértékek: {df['varos'].unique().tolist()}")

    return df


# ════════════════════════════════════════════════════════════
# 5. VIZUALIZÁCIÓ – BEFORE / AFTER
# ════════════════════════════════════════════════════════════

def visualizal_outlier(df_piszkos: pd.DataFrame, df_tiszta: pd.DataFrame, col: str) -> None:
    """Boxplot: piszkos vs. tiszta adat összehasonlítás.

    Args:
        df_piszkos: Eredeti DataFrame.
        df_tiszta:  Tisztított DataFrame.
        col:        Az összehasonlítandó oszlop.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Outlier kezelés: '{col}' – Előtte / Utána", fontsize=14)

    for ax, (adatok, cim) in zip(
        axes,
        [(df_piszkos[col], "⚠️  Eredeti (piszkos)"), (df_tiszta[col], "✅ Tisztított")],
    ):
        ax.boxplot(adatok.dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#4C72B0", alpha=0.6))
        ax.set_title(cim)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_outlier_before_after.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  📊 Grafikon mentve: outputs/02_outlier_before_after.png")


def visualizal_hianyzok(df: pd.DataFrame) -> None:
    """Hiányzó értékek hőtérképe.

    Args:
        df: DataFrame a vizualizációhoz.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        df.isnull().T,
        cmap="YlOrRd",
        cbar=False,
        ax=ax,
        yticklabels=True,
    )
    ax.set_title("Hiányzó értékek eloszlása (sárga = NaN)", fontsize=13)
    ax.set_xlabel("Sorok")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_hianyzok_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# FŐPROGRAM – TELJES CLEANING PIPELINE
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LECKE 02 – Adattisztítás")
    print("=" * 55)

    # 1. Piszkos adat betöltése
    df_raw = piszkos_adat_generalas(n=500)
    print(f"\n📥 Betöltött adat: {df_raw.shape}")

    # 2. Diagnózis
    diagnosztika(df_raw)

    # Hiányzó értékek vizualizálása
    visualizal_hianyzok(df_raw.head(100))

    # 3. Cleaning lépések
    df_clean = hianyzok_kezelese(df_raw)

    print("\n" + "=" * 55)
    print("🔧 OUTLIER KEZELÉS")
    print("=" * 55)
    df_clean, stat_bev = outlier_kezeles(df_clean, "bevetel", modszer="iqr")
    df_clean, stat_kor = outlier_kezeles(df_clean, "kor", modszer="iqr")

    df_clean = duplikat_es_tipushiba_kezelese(df_clean)

    # 4. Vizualizáció
    visualizal_outlier(df_raw, df_clean, "bevetel")

    # 5. Végeredmény
    print("\n" + "=" * 55)
    print("📋 VÉGEREDMÉNY")
    print("=" * 55)
    print(f"Eredeti sorok:   {len(df_raw)}")
    print(f"Tisztított sorok: {len(df_clean)}")
    print(f"Hiányzó értékek: {df_clean.isnull().sum().sum()}")

    # Mentés
    df_clean.to_csv(OUTPUT_DIR / "02_tiszta_adat.csv", index=False, encoding="utf-8")
    print("\n💾 Tiszta adat mentve: outputs/02_tiszta_adat.csv")

    print("\n✅ Lecke 02 sikeresen lefutott!")
    print("➡️  Következő: 08_ml_adatelokeszites.py")