"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 01 – Python Alapok Data Science szemszögből           ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - Python beépített adatstruktúrák magabiztos használata
  - List/dict comprehension és generátorok
  - Lambda, map, filter, zip
  - NumPy és Pandas első lépések
  - Típusannotáció (type hints) – modern Python stílus

ELŐFELTÉTEL: Python 3.10+, pip install numpy pandas
"""


from __future__ import annotations          # forward-reference type hints

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    # ── Importok ────────────────────────────────────────────────

    import math
    import random
    from collections import Counter, defaultdict
    from typing import Any

    import numpy as np
    import pandas as pd
except ImportError as _hiba:
    _csomag = str(_hiba).replace("No module named ", "").strip("'")
    _pip_map = {'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'Pillow', 'plotly': 'plotly', 'sns': 'seaborn', 'matplotlib': 'matplotlib', 'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'statsmodels': 'statsmodels', 'networkx': 'networkx', 'xgboost': 'xgboost', 'lightgbm': 'lightgbm', 'catboost': 'catboost', 'optuna': 'optuna', 'flaml': 'flaml', 'mlflow': 'mlflow', 'prophet': 'prophet', 'fastapi': 'fastapi', 'uvicorn': 'uvicorn', 'pydantic': 'pydantic', 'joblib': 'joblib', 'anthropic': 'anthropic', 'openai': 'openai', 'pingouin': 'pingouin', 'tqdm': 'tqdm'}
    _pip = _pip_map.get(_csomag.split(".")[0], _csomag.split(".")[0])
    print(f"\n\033[91m❌  Hiányzó csomag: {_csomag}\033[0m")
    print(f"\033[93m👉  Telepítsd: pip install {_pip}\033[0m")
    print("\033[96m💡  Vagy az összes egyszerre: pip install -r requirements.txt\033[0m\n")
    raise SystemExit(1)
# ────────────────────────────────────────────────────────────────



# ════════════════════════════════════════════════════════════
# 1. BEÉPÍTETT ADATSTRUKTÚRÁK
# ════════════════════════════════════════════════════════════

def demo_lista() -> None:
    """Lista műveletek – a leggyakoribb DS eszköz."""

    szamok: list[int] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

    print("=== LISTA ===")
    print(f"Eredeti:     {szamok}")
    print(f"Rendezett:   {sorted(szamok)}")               # nem módosítja az eredetit
    print(f"Egyedi:      {sorted(set(szamok))}")
    print(f"Leggyakoribb: {Counter(szamok).most_common(3)}")

    # Szeletelés (slicing) – DS-ben mindennapos
    print(f"Első 3 elem: {szamok[:3]}")
    print(f"Utolsó 3:    {szamok[-3:]}")
    print(f"Minden 2.:   {szamok[::2]}")


def demo_dict() -> None:
    """Szótár – rekordok és konfiguráció tárolása."""

    modell_metrikak: dict[str, float] = {
        "accuracy":  0.923,
        "precision": 0.891,
        "recall":    0.876,
        "f1":        0.883,
    }

    print("\n=== SZÓTÁR ===")
    for metrika, ertek in modell_metrikak.items():
        print(f"  {metrika:<12} {ertek:.3f}")

    # defaultdict – hiányzó kulcs esetén nem dob KeyError-t
    csoport_count: defaultdict[str, int] = defaultdict(int)
    kategoriak = ["A", "B", "A", "C", "B", "A"]
    for k in kategoriak:
        csoport_count[k] += 1
    print(f"\nCsoport-számok: {dict(csoport_count)}")


# ════════════════════════════════════════════════════════════
# 2. COMPREHENSION ÉS FUNKCIONÁLIS ESZKÖZÖK
# ════════════════════════════════════════════════════════════

def demo_comprehension() -> None:
    """List/dict/set comprehension – tömör, gyors kód."""

    print("\n=== COMPREHENSION ===")

    # List comprehension
    negyzetek = [x**2 for x in range(1, 11)]
    print(f"Négyzetek:    {negyzetek}")

    # Feltételes comprehension (szűrés)
    paros_negyzetek = [x**2 for x in range(1, 11) if x % 2 == 0]
    print(f"Páros²:       {paros_negyzetek}")

    # Dict comprehension – feature scaling kézi implementációja
    ertekek = {"kor": 34, "javedelem": 55000, "tapasztalat": 8}
    maximum = max(ertekek.values())
    normalt = {k: round(v / maximum, 4) for k, v in ertekek.items()}
    print(f"Normált:      {normalt}")

    # Set comprehension – egyedi karakterek
    szoveg = "data science python"
    egyedi_betuk = {c for c in szoveg if c != " "}
    print(f"Egyedi betűk: {sorted(egyedi_betuk)}")


def demo_lambda_map_filter() -> None:
    """Lambda, map, filter – funkcionális programozás."""

    print("\n=== LAMBDA / MAP / FILTER ===")

    adatok = [1.5, -2.3, 4.7, -0.5, 3.1, -1.8]

    # map – minden elemre alkalmaz egy függvényt
    abszolut = list(map(abs, adatok))
    print(f"Abszolút értékek: {abszolut}")

    # filter – csak a feltételnek megfelelők
    pozitiv = list(filter(lambda x: x > 0, adatok))
    print(f"Pozitívak:        {pozitiv}")

    # Lambda rendezési kulcsként
    szavak = ["pandas", "numpy", "sklearn", "plotly", "ai"]
    hossz_szerint = sorted(szavak, key=lambda s: len(s))
    print(f"Hossz szerint:    {hossz_szerint}")

    # zip – két lista összekapcsolása
    jellemzok = ["kor", "bevétel", "pontszám"]
    ertekek   = [29,    48000,     7.4]
    rekord = dict(zip(jellemzok, ertekek))
    print(f"Zip → dict:       {rekord}")


# ════════════════════════════════════════════════════════════
# 3. FÜGGVÉNYEK – TYPE HINTS ÉS DOCSTRING
# ════════════════════════════════════════════════════════════

def normalize(
    ertek: float,
    minimum: float,
    maximum: float,
) -> float:
    """Min-Max normalizáció egy értékre.

    Képlet: (x - min) / (max - min)

    Args:
        ertek:   A normalizálandó szám.
        minimum: Az adathalmaz minimuma.
        maximum: Az adathalmaz maximuma.

    Returns:
        0 és 1 közötti float érték.

    Raises:
        ValueError: Ha minimum == maximum (osztás nullával).
    """
    if minimum == maximum:
        raise ValueError("minimum és maximum nem lehet egyenlő!")
    return (ertek - minimum) / (maximum - minimum)


def describe_series(adatok: list[float]) -> dict[str, float]:
    """Leíró statisztika számítása listából.

    Args:
        adatok: Numerikus értékek listája.

    Returns:
        Szótár: mean, std, min, max, median kulcsokkal.
    """
    n = len(adatok)
    mean = sum(adatok) / n
    variance = sum((x - mean) ** 2 for x in adatok) / n
    std = math.sqrt(variance)
    sorted_data = sorted(adatok)
    mid = n // 2
    median = sorted_data[mid] if n % 2 else (sorted_data[mid - 1] + sorted_data[mid]) / 2

    return {
        "mean":   round(mean, 4),
        "std":    round(std, 4),
        "min":    min(adatok),
        "max":    max(adatok),
        "median": median,
    }


# ════════════════════════════════════════════════════════════
# 4. NUMPY ALAPOK
# ════════════════════════════════════════════════════════════

def demo_numpy() -> None:
    """NumPy – vektorizált számítások, sokkal gyorsabb mint sima Python."""

    print("\n=== NUMPY ===")

    rng = np.random.default_rng(seed=42)   # reprodukálható véletlen

    # 1D tömb létrehozása
    arr = rng.integers(1, 100, size=10)
    print(f"Array:    {arr}")
    print(f"Mean:     {arr.mean():.2f}")
    print(f"Std:      {arr.std():.2f}")
    print(f">50:      {arr[arr > 50]}")    # boolean indexelés

    # 2D tömb – mátrix műveletek
    matrix = rng.standard_normal((3, 4))   # 3 sor, 4 oszlop
    print(f"\nMátrix shape: {matrix.shape}")
    print(f"Soronkénti átlag: {matrix.mean(axis=1).round(3)}")
    print(f"Oszloponkénti max: {matrix.max(axis=0).round(3)}")

    # Broadcasting – skaláris műveletek vektorizálva
    celsius = np.array([0, 20, 37, 100])
    fahrenheit = celsius * 9/5 + 32
    print(f"\n°C: {celsius}")
    print(f"°F: {fahrenheit}")


# ════════════════════════════════════════════════════════════
# 5. PANDAS ALAPOK
# ════════════════════════════════════════════════════════════

def demo_pandas() -> None:
    """Pandas DataFrame – az adatelemzés alapköve."""

    print("\n=== PANDAS ===")

    # Szintetikus adathalmaz létrehozása
    rng = np.random.default_rng(seed=42)
    n = 100

    df = pd.DataFrame({
        "nev":        [f"Felhasználó_{i:03d}" for i in range(n)],
        "kor":        rng.integers(18, 65, n),
        "bevetel":    rng.integers(200_000, 1_500_000, n),   # HUF/hó
        "tapasztalat": rng.integers(0, 40, n),
        "kategoria":  rng.choice(["junior", "mid", "senior"], n),
        "aktiv":      rng.choice([True, False], n, p=[0.8, 0.2]),
    })

    print(f"Shape: {df.shape}")
    print(df.head(3).to_string())
    print(f"\nLeíró statisztika:")
    print(df[["kor", "bevetel", "tapasztalat"]].describe().round(1).to_string())

    # Szűrés, aggregálás
    print("\nKategóriánkénti átlagbér:")
    agg = (
        df.groupby("kategoria")["bevetel"]
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "átlag", "median": "medián", "count": "db"})
        .sort_values("átlag", ascending=False)
    )
    print(agg.to_string())

    # Method chaining – modern Pandas stílus
    top_senior = (
        df
        .query("kategoria == 'senior' and aktiv == True")
        .sort_values("bevetel", ascending=False)
        .head(5)
        [["nev", "kor", "bevetel"]]
    )
    print(f"\nTop 5 aktív senior (bevétel alapján):")
    print(top_senior.to_string(index=False))


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  LECKE 01 – Python Alapok")
    print("=" * 60)

    demo_lista()
    demo_dict()
    demo_comprehension()
    demo_lambda_map_filter()

    # Saját függvény tesztelése
    print("\n=== SAJÁT FÜGGVÉNYEK ===")
    adatok = [random.gauss(50, 15) for _ in range(200)]
    stats = describe_series(adatok)
    print(f"Leíró statisztika: {stats}")
    print(f"Normált 75: {normalize(75, min(adatok), max(adatok)):.4f}")

    demo_numpy()
    demo_pandas()

    print("\n✅ Lecke 01 sikeresen lefutott!")
    print("➡️  Következő: 02_adattisztitas.py")