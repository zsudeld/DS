"""
24 – SQL & Adatelérés
=====================
Célok:
  - SQLite adatbázis létrehozása és lekérdezése Pythonból
  - Pandas + SQL integráció (read_sql, to_sql)
  - Legfontosabb SQL parancsok adattudomány szemszögéből
  - Valós workflow: adatbázisból → DataFrame → elemzés

Miért SQL?
  A legtöbb vállalatnál az adatok adatbázisban vannak (MySQL, PostgreSQL,
  BigQuery, Snowflake stb.). A data scientist munkájának ~50%-a adatlekérdezés.
  SQLite = ugyanaz a szintaxis, fájlként fut, telepítés nélkül.
"""

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import sqlite3
    import pandas as pd
    import numpy as np
    from pathlib import Path
except ImportError as _hiba:
    _csomag = str(_hiba).replace("No module named ", "").strip("'")
    _pip_map = {'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'Pillow', 'plotly': 'plotly', 'sns': 'seaborn', 'matplotlib': 'matplotlib', 'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'statsmodels': 'statsmodels', 'networkx': 'networkx', 'xgboost': 'xgboost', 'lightgbm': 'lightgbm', 'catboost': 'catboost', 'optuna': 'optuna', 'flaml': 'flaml', 'mlflow': 'mlflow', 'prophet': 'prophet', 'fastapi': 'fastapi', 'uvicorn': 'uvicorn', 'pydantic': 'pydantic', 'joblib': 'joblib', 'anthropic': 'anthropic', 'openai': 'openai', 'pingouin': 'pingouin', 'tqdm': 'tqdm'}
    _pip = _pip_map.get(_csomag.split(".")[0], _csomag.split(".")[0])
    print(f"\n\033[91m❌  Hiányzó csomag: {_csomag}\033[0m")
    print(f"\033[93m👉  Telepítsd: pip install {_pip}\033[0m")
    print("\033[96m💡  Vagy az összes egyszerre: pip install -r requirements.txt\033[0m\n")
    raise SystemExit(1)
# ────────────────────────────────────────────────────────────────


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = OUTPUT_DIR / "kurzus_demo.db"


# ─────────────────────────────────────────────────────────────────────────────
# 1. ADATBÁZIS LÉTREHOZÁSA – Szintetikus vállalati adatok
# ─────────────────────────────────────────────────────────────────────────────

def adatbazis_letrehozasa():
    """
    3 táblát hozunk létre:
      - alkalmazottak: HR adatok
      - ertekesites:   havi értékesítési rekordok
      - termekek:      termékkatalogus

    Kapcsolatok (idegen kulcsok):
      ertekesites.alkalmazott_id → alkalmazottak.id
      ertekesites.termek_id      → termekek.id
    """
    print("\n=== 1. Adatbázis létrehozása ===")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tábla törlése ha már létezik (újrafuttathatóság)
    cursor.executescript("""
        DROP TABLE IF EXISTS ertekesites;
        DROP TABLE IF EXISTS alkalmazottak;
        DROP TABLE IF EXISTS termekek;
    """)

    # ── alkalmazottak tábla ────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE alkalmazottak (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nev         TEXT NOT NULL,
            osztaly     TEXT NOT NULL,
            fizetes     INTEGER NOT NULL,
            felveve      DATE NOT NULL,
            varos       TEXT
        )
    """)

    # ── termekek tábla ─────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE termekek (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nev         TEXT NOT NULL,
            kategoria   TEXT NOT NULL,
            ar          REAL NOT NULL,
            keszlet     INTEGER DEFAULT 0
        )
    """)

    # ── ertekesites tábla ──────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE ertekesites (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            alkalmazott_id  INTEGER NOT NULL,
            termek_id       INTEGER NOT NULL,
            datum           DATE NOT NULL,
            mennyiseg       INTEGER NOT NULL,
            engedmeny_pct   REAL DEFAULT 0,
            FOREIGN KEY (alkalmazott_id) REFERENCES alkalmazottak(id),
            FOREIGN KEY (termek_id)      REFERENCES termekek(id)
        )
    """)

    # ── Adatok feltöltése ──────────────────────────────────────────────────
    np.random.seed(42)

    alkalmazottak_adatok = [
        ("Kovács Péter", "Sales", 450_000, "2020-03-15", "Budapest"),
        ("Nagy Eszter",  "Sales", 520_000, "2019-07-01", "Debrecen"),
        ("Tóth Gábor",   "IT",    680_000, "2021-01-10", "Budapest"),
        ("Varga Ildikó", "Sales", 490_000, "2018-11-20", "Győr"),
        ("Fekete Norbert","Marketing",410_000,"2022-05-05","Budapest"),
        ("Szabó Anna",   "Sales", 540_000, "2017-09-12", "Pécs"),
    ]
    cursor.executemany(
        "INSERT INTO alkalmazottak (nev, osztaly, fizetes, felveve, varos) VALUES (?,?,?,?,?)",
        alkalmazottak_adatok
    )

    termek_adatok = [
        ("Laptop Pro",    "Elektronika", 450_000, 25),
        ("Monitor 27\"",  "Elektronika", 125_000, 40),
        ("Szék ergonomikus","Bútor",      85_000, 60),
        ("Headset BT",    "Elektronika",  35_000, 80),
        ("Íróasztal",     "Bútor",        72_000, 30),
        ("Tablet 10\"",   "Elektronika", 185_000, 45),
    ]
    cursor.executemany(
        "INSERT INTO termekek (nev, kategoria, ar, keszlet) VALUES (?,?,?,?)",
        termek_adatok
    )

    # Értékesítési rekordok generálása
    ertekesites_adatok = []
    for _ in range(200):
        alk_id = np.random.randint(1, 7)
        ter_id = np.random.randint(1, 7)
        ev = np.random.randint(2022, 2026)
        ho = np.random.randint(1, 13)
        nap = np.random.randint(1, 29)
        datum = f"{ev}-{ho:02d}-{nap:02d}"
        mennyiseg = np.random.randint(1, 6)
        engedmeny = np.random.choice([0, 5, 10, 15], p=[0.5, 0.25, 0.15, 0.10])
        # int() / float() konverzió: numpy típusok SQLite-ban bytes-ként tárolódhatnak
        ertekesites_adatok.append((int(alk_id), int(ter_id), datum,
                                   int(mennyiseg), float(engedmeny)))

    cursor.executemany(
        "INSERT INTO ertekesites (alkalmazott_id, termek_id, datum, mennyiseg, engedmeny_pct) VALUES (?,?,?,?,?)",
        ertekesites_adatok
    )

    conn.commit()
    conn.close()
    print(f"  ✅ Adatbázis létrehozva: {DB_PATH}")
    print(f"     Táblák: alkalmazottak ({len(alkalmazottak_adatok)} sor), "
          f"termekek ({len(termek_adatok)} sor), "
          f"ertekesites ({len(ertekesites_adatok)} sor)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ALAP SQL LEKÉRDEZÉSEK
# ─────────────────────────────────────────────────────────────────────────────

def alap_lekerdezesek():
    """
    A data scientist leggyakrabban használt SQL parancsai:
      SELECT, WHERE, ORDER BY, LIMIT, DISTINCT
    """
    print("\n=== 2. Alap SQL lekérdezések ===")

    conn = sqlite3.connect(DB_PATH)

    # SELECT *  – mindent lekér
    print("\n  [a] Összes alkalmazott:")
    df = pd.read_sql("SELECT * FROM alkalmazottak", conn)
    print(df.to_string(index=False))

    # WHERE – szűrés
    print("\n  [b] Sales-esek, akik 500.000 Ft felett keresnek:")
    df = pd.read_sql("""
        SELECT nev, fizetes, varos
        FROM   alkalmazottak
        WHERE  osztaly = 'Sales'
          AND  fizetes > 500000
        ORDER  BY fizetes DESC
    """, conn)
    print(df.to_string(index=False))

    # DISTINCT – egyedi értékek
    print("\n  [c] Egyedi városok:")
    df = pd.read_sql("SELECT DISTINCT varos FROM alkalmazottak ORDER BY varos", conn)
    print(df.to_string(index=False))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. AGGREGÁCIÓ ÉS CSOPORTOSÍTÁS
# ─────────────────────────────────────────────────────────────────────────────

def aggregacio():
    """
    GROUP BY + aggregáló függvények (COUNT, SUM, AVG, MIN, MAX)
    HAVING = WHERE, de GROUP BY UTÁN szűr
    """
    print("\n=== 3. Aggregáció és csoportosítás ===")

    conn = sqlite3.connect(DB_PATH)

    # Átlagfizetés osztályonként
    print("\n  [a] Átlagfizetés osztályonként:")
    df = pd.read_sql("""
        SELECT   osztaly,
                 COUNT(*)            AS letszam,
                 AVG(fizetes)        AS atlag_fizetes,
                 MIN(fizetes)        AS min_fizetes,
                 MAX(fizetes)        AS max_fizetes
        FROM     alkalmazottak
        GROUP BY osztaly
        ORDER BY atlag_fizetes DESC
    """, conn)
    df["atlag_fizetes"] = df["atlag_fizetes"].round(0).astype(int)
    print(df.to_string(index=False))

    # Kategóriánkénti bevétel (engedménnyel korrigálva)
    print("\n  [b] Termék-kategóriánkénti bevétel (top 3):")
    df = pd.read_sql("""
        SELECT   t.kategoria,
                 COUNT(*)                                       AS tranzakciok,
                 SUM(e.mennyiseg)                               AS ossz_db,
                 ROUND(SUM(e.mennyiseg * t.ar
                       * (1 - e.engedmeny_pct / 100.0)), 0)    AS netto_bevetel
        FROM     ertekesites e
        JOIN     termekek    t ON t.id = e.termek_id
        GROUP BY t.kategoria
        HAVING   ossz_db > 5
        ORDER BY netto_bevetel DESC
        LIMIT 3
    """, conn)
    print(df.to_string(index=False))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. JOIN – Táblák összekapcsolása
# ─────────────────────────────────────────────────────────────────────────────

def join_muveletek():
    """
    JOIN típusok:
      INNER JOIN: csak a mindkét táblában lévő sorok
      LEFT JOIN:  bal tábla összes sora + ahol van találat a jobb táblából
      (RIGHT/FULL JOIN SQLite-ban nem támogatott, de más DB-kben igen)

    Aranyszabály: mindig azt a JOIN-t válaszd, amelyik a kérdésednek megfelel.
      "Minden alkalmazott, akinek volt értékesítése" → INNER JOIN
      "Minden alkalmazott, még ha nem volt is értékesítése" → LEFT JOIN
    """
    print("\n=== 4. JOIN műveletek ===")

    conn = sqlite3.connect(DB_PATH)

    print("\n  [a] Értékesítők teljesítménye (INNER JOIN):")
    df = pd.read_sql("""
        SELECT   a.nev,
                 a.osztaly,
                 COUNT(e.id)                             AS ertekesitesek,
                 SUM(e.mennyiseg)                        AS ossz_db,
                 ROUND(AVG(e.engedmeny_pct), 1)          AS atlag_engedmeny
        FROM     alkalmazottak a
        INNER JOIN ertekesites e ON e.alkalmazott_id = a.id
        GROUP BY a.id, a.nev, a.osztaly
        ORDER BY ertekesitesek DESC
    """, conn)
    print(df.to_string(index=False))

    print("\n  [b] Minden alkalmazott – volt-e értékesítése? (LEFT JOIN):")
    df = pd.read_sql("""
        SELECT   a.nev,
                 a.osztaly,
                 COALESCE(COUNT(e.id), 0) AS ertekesitesek
        FROM     alkalmazottak a
        LEFT JOIN ertekesites e ON e.alkalmazott_id = a.id
        GROUP BY a.id, a.nev, a.osztaly
        ORDER BY ertekesitesek DESC
    """, conn)
    print(df.to_string(index=False))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. ABLAKFÜGGVÉNYEK (Window Functions) – Haladó SQL
# ─────────────────────────────────────────────────────────────────────────────

def ablakfuggvenyek():
    """
    Ablakfüggvények: aggregálnak, de NEM tömörítik a sorokat (nem kell GROUP BY).
    Nagyon hasznos:
      - Rangsorolás (RANK, ROW_NUMBER, DENSE_RANK)
      - Gördülő összegek / átlagok (SUM ... OVER ...)
      - Előző sor értéke (LAG), következő sor (LEAD)
    """
    print("\n=== 5. Ablakfüggvények (Window Functions) ===")

    conn = sqlite3.connect(DB_PATH)

    print("\n  [a] Értékesítők rangsorolása:")
    df = pd.read_sql("""
        SELECT   a.nev,
                 COUNT(e.id)   AS ertekesitesek,
                 RANK() OVER (ORDER BY COUNT(e.id) DESC) AS rang
        FROM     alkalmazottak a
        JOIN     ertekesites   e ON e.alkalmazott_id = a.id
        GROUP BY a.id, a.nev
        ORDER BY rang
    """, conn)
    print(df.to_string(index=False))

    print("\n  [b] Havi bevétel + gördülő összeg:")
    df = pd.read_sql("""
        SELECT   strftime('%Y-%m', datum)          AS ev_ho,
                 ROUND(SUM(e.mennyiseg * t.ar), 0) AS ho_bevetel,
                 ROUND(SUM(SUM(e.mennyiseg * t.ar))
                       OVER (ORDER BY strftime('%Y-%m', datum)), 0) AS gordulo_osszeg
        FROM     ertekesites e
        JOIN     termekek    t ON t.id = e.termek_id
        GROUP BY ev_ho
        ORDER BY ev_ho
        LIMIT 12
    """, conn)
    print(df.to_string(index=False))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6. PANDAS ↔ SQL INTEGRÁCIÓ
# ─────────────────────────────────────────────────────────────────────────────

def pandas_sql_integracios():
    """
    A valós workflow:
      1. read_sql() → SQL lekérdezés eredménye DataFrame-be
      2. DataFrame manipuláció Pandas-szal
      3. to_sql() → eredmény visszaírása adatbázisba (riport tábla)
    """
    print("\n=== 6. Pandas ↔ SQL integráció ===")

    conn = sqlite3.connect(DB_PATH)

    # Lekérdezés DataFrame-be
    df = pd.read_sql("""
        SELECT e.datum, a.nev, t.nev AS termek, t.kategoria,
               e.mennyiseg, t.ar, e.engedmeny_pct,
               ROUND(e.mennyiseg * t.ar * (1 - e.engedmeny_pct/100.0), 0) AS netto
        FROM   ertekesites e
        JOIN   alkalmazottak a ON a.id = e.alkalmazott_id
        JOIN   termekek      t ON t.id = e.termek_id
    """, conn, parse_dates=["datum"])

    print(f"  Lekért {len(df)} sor. Első 3:")
    print(df.head(3).to_string(index=False))

    # Pandas transzformáció
    riport = (df.groupby(df["datum"].dt.to_period("M"))
                .agg(tranzakciok=("netto", "count"),
                     ossz_bevetel=("netto", "sum"),
                     atlag_engedmeny=("engedmeny_pct", "mean"))
                .reset_index()
                .rename(columns={"datum": "idoszak"}))
    riport["idoszak"] = riport["idoszak"].astype(str)
    riport["ossz_bevetel"] = riport["ossz_bevetel"].round(0).astype(int)
    riport["atlag_engedmeny"] = riport["atlag_engedmeny"].round(1)

    # Visszaírás adatbázisba
    riport.to_sql("havi_riport", conn, if_exists="replace", index=False)
    print(f"\n  ✅ 'havi_riport' tábla mentve az adatbázisba ({len(riport)} sor)")
    print(riport.tail(5).to_string(index=False))

    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# FŐPROGRAM
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  24 – SQL & Adatelérés")
    print("=" * 60)

    adatbazis_letrehozasa()
    alap_lekerdezesek()
    aggregacio()
    join_muveletek()
    ablakfuggvenyek()
    pandas_sql_integracios()

    print("\n✅ Kész!")
    print("\nKövetkező lépés: 25_idosor_elorejelzes.py – Idősor-előrejelzés")