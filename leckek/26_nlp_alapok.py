"""
26 – NLP alapok: Szövegfeldolgozás
====================================
Célok:
  - Szöveg előkészítés (tokenizáció, stopszavak, normalizálás)
  - Bag of Words és TF-IDF vektorizáció
  - Szövegklasszifikáció ML modellel
  - Word2Vec / szóbeágyazás alapfogalmak
  - Mikor NLP vs. LLM API?

Mikor NLP vs. LLM API (03_ai_databiz.py)?
  → Klasszikus NLP: nagy volumen, alacsony latencia, nincs API-költség,
    egyszerű feladatok (spam, sentiment, kategorizálás)
  → LLM API:        komplex megértés, indoklás, kevés tanítóadat, kreatív szöveg
"""

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import re
    import warnings
    warnings.filterwarnings("ignore")
    from pathlib import Path
    from collections import Counter
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. SZINTETIKUS SZÖVEGES ADATHALMAZ
# ─────────────────────────────────────────────────────────────────────────────

def adathalmaz_generalas() -> pd.DataFrame:
    """
    Szintetikus ügyfélvélemény (review) adathalmaz.
    Kategóriák: pozitív, negatív, semleges
    """
    pozitiv = [
        "Kiváló termék, nagyon meg vagyok elégedve!",
        "Gyors szállítás, a csomag tökéletes állapotban érkezett.",
        "Remek minőség, az árhoz képest fantasztikus.",
        "Nagyon ajánlom mindenkinek, teljesen elégedett vagyok.",
        "Pontosan olyan, mint amit vártam, jó vétel volt.",
        "Szuper szolgáltatás, visszajövök biztosan.",
        "Minőségi termék, gyors kiszállítás. Köszönöm!",
        "Tökéletes! Mindent megkaptam, amit ígértek.",
        "Az ügyfélszolgálat is segítőkész volt. Elégedett vagyok.",
        "Nagyon szép kivitelezés, megérte az árát.",
    ]
    negativ = [
        "Rossz minőség, két hét után tönkrement.",
        "A csomag sérülten érkezett, nagyon csalódott vagyok.",
        "Nem azt kaptam, amit rendeltem. Visszaküldöm.",
        "Lassú szállítás és az ügyfélszolgálat nem segített.",
        "Egyáltalán nem ajánlom, teljes pénzkidobás volt.",
        "Olcsó anyag, rögtön szakadt. Szégyen.",
        "Nem működött megfelelően, vissza kellett küldeni.",
        "Nagyon rossz tapasztalat, nem rendelek többet innen.",
        "Sérült terméket küldtek, reklamáltam de nem reagáltak.",
        "Az ár és a minőség nem arányos. Csalódtam.",
    ]
    semleges = [
        "Megkaptam a terméket, most próbálom ki.",
        "Az átlagnak megfelel, semmi különös.",
        "Rendben van, de volt jobb is már.",
        "Nem rossz, de nem is kiemelkedő.",
        "Hozza az elvártat, nincs más mondandóm.",
        "Standard minőség, elvárt volt.",
        "Kaptam amit rendeltem, a többi mindegy.",
        "Normális termék, közepes minőség.",
        "Egyszerű, de célt ér.",
        "Megfelel az alapvető igényeknek.",
    ]

    szovegek = pozitiv + negativ + semleges
    cimkek = (["pozitiv"] * len(pozitiv) +
              ["negativ"] * len(negativ) +
              ["semleges"] * len(semleges))

    df = pd.DataFrame({"szoveg": szovegek, "cimke": cimkek})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SZÖVEG ELŐKÉSZÍTÉS
# ─────────────────────────────────────────────────────────────────────────────

def szoveg_elokeszites(szoveg: str, stopszavak: set = None) -> list[str]:
    """
    Szöveg előfeldolgozás lépései:
      1. Kisbetűsítés (lowercasing)
      2. Írásjelek és számok eltávolítása
      3. Tokenizáció (szavakra bontás)
      4. Stopszavak eltávolítása (pl. 'és', 'az', 'a')
      5. (Opcionális) Stemming/Lemmatizálás – tövet keresünk

    Magyar stopszavak: rövid lista, valós projektben bővebb szükséges.
    """
    if stopszavak is None:
        stopszavak = {
            "a", "az", "és", "is", "de", "ha", "hogy", "nem", "volt",
            "van", "egy", "ezt", "azt", "ez", "el", "fel", "meg", "ki",
            "be", "le", "mint", "már", "még", "csak", "is", "sem",
        }

    # 1. Kisbetűsítés
    szoveg = szoveg.lower()
    # 2. Írásjelek eltávolítása (csak betűk maradnak, ékezetes is)
    szoveg = re.sub(r"[^a-záéíóöőúüű\s]", " ", szoveg)
    # 3. Tokenizáció
    tokenek = szoveg.split()
    # 4. Stopszavak eltávolítása + rövid szavak kiszűrése
    tokenek = [t for t in tokenek if t not in stopszavak and len(t) > 2]
    return tokenek


def elokeszites_demo(df: pd.DataFrame):
    print("\n=== 2. Szöveg előkészítés ===")
    print("\n  Eredeti → Tokenek (első 5 sor):")
    for _, sor in df.head(5).iterrows():
        tokenek = szoveg_elokeszites(sor["szoveg"])
        print(f"  [{sor['cimke']}] '{sor['szoveg'][:50]}...'")
        print(f"        Tokenek: {tokenek}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BAG OF WORDS ÉS TF-IDF
# ─────────────────────────────────────────────────────────────────────────────

def tfidf_vektorizacio(df: pd.DataFrame):
    """
    Bag of Words (BoW):
      Minden szó egy jellemző, értéke a szó előfordulásainak száma.
      Probléma: a 'van', 'egy' is nagy súlyt kap → stopszavak szükségesek.

    TF-IDF (Term Frequency – Inverse Document Frequency):
      TF  = szó gyakorisága az adott dokumentumban
      IDF = ritka szó → nagyobb súly; minden dokumentumban lévő szó → kis súly
      Eredmény: a dokumentumra jellemző, fontos szavak kapnak nagy értéket.
    """
    print("\n=== 3. TF-IDF vektorizáció ===")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    except ImportError:
        print("  [!] sklearn nem elérhető")
        return None, None

    # Tokenek összefűzése (sklearn szöveget vár, nem listát)
    def elokeszit(szoveg):
        return " ".join(szoveg_elokeszites(szoveg))

    szovegek_clean = df["szoveg"].apply(elokeszit)

    # TF-IDF mátrix
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2))  # Bigramok is
    X = tfidf.fit_transform(szovegek_clean)

    print(f"  Dokumentumok: {X.shape[0]}, Jellemzők: {X.shape[1]}")

    # Top szavak kategóriánként
    szavak = tfidf.get_feature_names_out()
    df_tfidf = pd.DataFrame(X.toarray(), columns=szavak)
    df_tfidf["cimke"] = df["cimke"].values

    print("\n  Top 5 TF-IDF szó kategóriánként:")
    for cimke in ["pozitiv", "negativ", "semleges"]:
        subset = df_tfidf[df_tfidf["cimke"] == cimke].drop("cimke", axis=1)
        top5 = subset.mean().sort_values(ascending=False).head(5)
        print(f"    [{cimke}]: {list(top5.index)}")

    return X, tfidf, szovegek_clean


# ─────────────────────────────────────────────────────────────────────────────
# 4. SZÖVEGKLASSZIFIKÁCIÓ
# ─────────────────────────────────────────────────────────────────────────────

def szovegklasszifikacio(df: pd.DataFrame, X, szovegek_clean):
    """
    Klasszikus ML pipeline szöveghez:
      TfidfVectorizer → LogisticRegression / LinearSVC / MultinomialNB

    Logisztikus regresszió ajánlott kiindulópontként:
      - Gyors
      - Értelmezhető együtthatók (melyik szó melyik kategóriát jelzi)
      - Általában versenyképes eredményt ad
    """
    print("\n=== 4. Szövegklasszifikáció ===")

    try:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.svm import LinearSVC
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
    except ImportError:
        print("  [!] sklearn nem elérhető")
        return

    y = df["cimke"]

    # Kis adat → cross-validation megbízhatóbb, mint egyetlen train/test split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modellek = {
        "Logisztikus Regresszió": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "Naív Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100)),
            ("clf", LinearSVC(max_iter=2000, C=1.0)),
        ]),
    }

    print("\n  Modellek összehasonlítása (5-fold CV):")
    legjobb_modell = None
    legjobb_score = 0

    for nev, pipeline in modellek.items():
        scores = cross_val_score(pipeline, szovegek_clean, y, cv=cv,
                                 scoring="accuracy")
        print(f"    {nev:<28}: {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > legjobb_score:
            legjobb_score = scores.mean()
            legjobb_modell = (nev, pipeline)

    # Legjobb modell részletes kiértékelése
    nev, pipeline = legjobb_modell
    print(f"\n  ✅ Legjobb modell: {nev}")

    # Train az összes adaton, kiértékelés train adaton (csak illusztrációként)
    pipeline.fit(szovegek_clean, y)
    y_pred = pipeline.predict(szovegek_clean)
    print("\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=["negativ", "pozitiv", "semleges"]))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=["pozitiv", "negativ", "semleges"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["pozitiv", "negativ", "semleges"],
                yticklabels=["pozitiv", "negativ", "semleges"], ax=ax)
    ax.set_title(f"Confusion Matrix – {nev}", fontweight="bold")
    ax.set_xlabel("Előrejelzett")
    ax.set_ylabel("Valós")
    plt.tight_layout()
    fpath = OUTPUT_DIR / "26a_nlp_confusion_matrix.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Confusion matrix mentve: {fpath}")

    # Jellemző fontossági szavak (Logisztikus regresszióból)
    if "Logisztikus" in nev:
        _jellemzo_szavak_plot(pipeline)

    return pipeline


def _jellemzo_szavak_plot(pipeline):
    """Top szavak vizualizálása osztályonként."""
    try:
        import seaborn as sns
        tfidf_vec = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        szavak = tfidf_vec.get_feature_names_out()
        osztalyok = clf.classes_

        fig, axes = plt.subplots(1, len(osztalyok), figsize=(15, 5))
        for i, (osztaly, ax) in enumerate(zip(osztalyok, axes)):
            koefficiensek = clf.coef_[i]
            top_idx = np.argsort(koefficiensek)[-10:]
            top_szavak = szavak[top_idx]
            top_koeff = koefficiensek[top_idx]
            ax.barh(top_szavak, top_koeff, color=sns.color_palette("husl", 3)[i])
            ax.set_title(f"'{osztaly}' kategória\nTop 10 szó", fontweight="bold")
            ax.set_xlabel("Koefficiens értéke")

        plt.suptitle("Logisztikus Regresszió – Fontos szavak kategóriánként",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        fpath = OUTPUT_DIR / "26b_nlp_fontos_szavak.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Fontos szavak mentve: {fpath}")
    except Exception as e:
        print(f"  [!] Szó-plot hiba: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SZÓBEÁGYAZÁS ALAPFOGALMAK (Word2Vec / GloVe)
# ─────────────────────────────────────────────────────────────────────────────

def szobeagyazas_fogalmak():
    """
    Word2Vec / GloVe / FastText – szóbeágyazás elméleti összefoglaló.

    A TF-IDF problémája: 'király' és 'monarcha' teljesen különböző vektor,
    bár jelentésük közel azonos. A szóbeágyazás ezt oldja meg.

    Szóbeágyazás lényege:
      Minden szó → sűrű numerikus vektor (pl. 300 dimenzió)
      Hasonló szavak → közel vannak a vektortérben (koszinusz-hasonlóság)

    Híres analógia:
      király - férfi + nő ≈ királynő

    Elérhetőség:
      - gensim: Word2Vec, FastText tanítása
      - spacy: előtanított modellek (hu_core_news_sm magyar modell)
      - HuggingFace transformers: BERT, RoBERTa (kontextuális beágyazás)

    Mikor érdemes szóbeágyazást használni?
      - TF-IDF nem elég (szemantikai hasonlóság is számít)
      - Szövegek hasonlóság-alapú keresése (semantic search)
      - Kis tanítóadat (pretrained vektor → jobb generalizáció)

    LLM vs. klasszikus beágyazás:
      → Sentence-transformers (all-MiniLM-L6-v2): gyors, ingyenes, helyi futás
      → OpenAI text-embedding-3-small: legjobb minőség, API-költség
    """
    print("\n=== 5. Szóbeágyazás (Word2Vec/GloVe) – fogalmak ===")
    print("""
  Szóbeágyazás: minden szó → numerikus vektor
  ─────────────────────────────────────────────
  TF-IDF vektor (ritka):   [0, 0, 1.2, 0, 0, 0.8, 0, ...]
  Word2Vec vektor (sűrű):  [0.23, -0.15, 0.87, 0.41, ...]  (300 dim)

  Hasonlóság mérés: koszinusz-hasonlóság
    cos(A, B) = A·B / (|A|×|B|)  →  1.0 = teljesen hasonló, 0 = független

  Magyar NLP eszközök:
    - spacy + hu_core_news_sm:   gyors, ingyenes, alapfeladatok
    - HuggingFace BERT (multilingual): legjobb minőség
    - sentence-transformers:     szemantikus keresés

  Könyvtárak telepítése:
    pip install gensim spacy sentence-transformers
    python -m spacy download hu_core_news_sm
    """)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ÚJ SZÖVEG OSZTÁLYOZÁSA
# ─────────────────────────────────────────────────────────────────────────────

def uj_szoveg_osztalyozasa(pipeline):
    """Betanított modell kipróbálása új szövegeken."""
    print("\n=== 6. Új szövegek osztályozása ===")

    uj_szovegek = [
        "Fantasztikus vásárlási élmény, mindent ajánlok!",
        "Teljesen hasznavehetetlen termék, nem működött.",
        "Megkaptam, normális, semmi extra.",
        "Nagyon gyors kiszállítás és barátságos ügyfélszolgálat.",
        "Sajnos a méret nem stimmel, visszaküldöm.",
    ]

    import re as _re
    STOPSZAVAK = {
        "a", "az", "és", "is", "de", "ha", "hogy", "nem", "volt",
        "van", "egy", "ezt", "azt", "ez", "el", "fel", "meg", "ki",
        "be", "le", "mint", "már", "még", "csak", "is", "sem",
    }

    def elokeszit(szoveg):
        szoveg = szoveg.lower()
        szoveg = _re.sub(r"[^a-záéíóöőúüű\s]", " ", szoveg)
        tokenek = [t for t in szoveg.split() if t not in STOPSZAVAK and len(t) > 2]
        return " ".join(tokenek)

    clean = [elokeszit(s) for s in uj_szovegek]
    joslatok = pipeline.predict(clean)

    EMOJI = {"pozitiv": "😊", "negativ": "😠", "semleges": "😐"}
    print()
    for szoveg, joslatom in zip(uj_szovegek, joslatok):
        print(f"  {EMOJI[joslatom]} [{joslatom:>8}]  \"{szoveg}\"")


# ─────────────────────────────────────────────────────────────────────────────
# FŐPROGRAM
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  26 – NLP alapok: Szövegfeldolgozás")
    print("=" * 60)

    df = adathalmaz_generalas()
    print(f"\n  Adathalmaz: {len(df)} szöveg, {df['cimke'].value_counts().to_dict()}")

    elokeszites_demo(df)

    eredmeny = tfidf_vektorizacio(df)
    if eredmeny is not None:
        X, tfidf_obj, szovegek_clean = eredmeny
        pipeline = szovegklasszifikacio(df, X, szovegek_clean)
        if pipeline is not None:
            uj_szoveg_osztalyozasa(pipeline)

    szobeagyazas_fogalmak()

    print("\n✅ Kész!")
    print("\nKövetkező lépés: 27_deployment_mlops.py – Modell élesbe helyezése")