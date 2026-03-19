"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 04 – AI Onboarding Esettanulmány                      ║
║  LECKE 05 – Kriptovaluta Bot                                 ║
║  LECKE 06 – Random Network Elemzés                           ║
╚══════════════════════════════════════════════════════════════╝
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import warnings; warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import os
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
# LECKE 04 – AI ONBOARDING ESETTANULMÁNY
# ════════════════════════════════════════════════════════════

def ai_onboarding_esettanulmany() -> None:
    """HR Onboarding sikeres befejezés előrejelzése.

    ÜZLETI PROBLÉMA:
      Egy 500 fős cég évenként ~80 új dolgozót vesz fel.
      A beilleszkedési programot ~30% nem fejezi be sikeresen.
      Kérdés: Melyik jelöltek vannak veszélyben? → Proaktív segítség.

    MEGKÖZELÍTÉS:
      1. Onboarding adatok összegyűjtése (HR rendszer)
      2. Bináris osztályozás: Befejezi-e az onboardingot?
      3. Modell: Random Forest (interpretálható, HR-nek érthető)
      4. SHAP értékek a magyarázathoz
      5. Kockázati szegmentálás
    """
    print("=" * 60)
    print("  LECKE 04 – AI Onboarding Esettanulmány")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # ── Adathalmaz generálása ────────────────────────────────
    rng = np.random.default_rng(42)
    n = 600

    kor            = rng.integers(22, 55, n).astype(float)
    tapasztalat_ev = rng.integers(0, 20, n).astype(float)
    elso_het_login = rng.integers(0, 20, n).astype(float)     # bejelentkezések száma
    mentori_talalko = rng.integers(0, 10, n).astype(float)    # mentori találkozók
    kepzesi_teljesit = rng.uniform(0, 100, n)                  # %
    csapat_meret    = rng.integers(3, 20, n).astype(float)
    tavoli_munka    = rng.choice([0, 1], n, p=[0.6, 0.4]).astype(float)

    # Célváltozó: sikeresen befejezi az onboardingot
    # (magasabb login, mentori találkozó → nagyobb esély)
    logit = (
        0.15 * elso_het_login
        + 0.3  * mentori_talalko
        + 0.02 * kepzesi_teljesit
        - 0.1  * tavoli_munka
        - 0.05 * tapasztalat_ev
        + rng.normal(0, 1.5, n)
    )
    cel = (logit > 1.5).astype(int)

    df = pd.DataFrame({
        "kor": kor, "tapasztalat_ev": tapasztalat_ev,
        "elso_het_login": elso_het_login, "mentori_talalko": mentori_talalko,
        "kepzesi_teljesit": kepzesi_teljesit, "csapat_meret": csapat_meret,
        "tavoli_munka": tavoli_munka, "siker": cel,
    })

    print(f"\nAdathalmaz: {df.shape}")
    print(f"Siker arány: {cel.mean():.1%}")

    # ── Modell tanítása ──────────────────────────────────────
    X = df.drop("siker", axis=1)
    y = df["siker"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(classification_report(y_test, rf.predict(X_test), target_names=["Kockázatos", "Sikeres"]))

    # ── Kockázati szegmentálás ───────────────────────────────
    df_test = X_test.copy()
    df_test["kockaz_pontszam"] = y_proba
    df_test["kockaz_kategoria"] = pd.cut(
        y_proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Magas kockázat", "Közepes", "Alacsony"],
    )
    print(f"\nKockázati szegmensek:\n{df_test['kockaz_kategoria'].value_counts().to_string()}")

    # ── Feature importance vizualizálás ─────────────────────
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", ax=ax, color=["#E74C3C" if v > imp.median() else "#BDC3C7" for v in imp])
    ax.set_title("Onboarding siker előrejelzői (RF Feature Importance)")
    ax.axvline(imp.median(), color="navy", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_onboarding_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── SHAP magyarázat (ha telepítve) ───────────────────────
    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(X_test.iloc[:50])
        shap.summary_plot(shap_vals[1], X_test.iloc[:50], show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("  ✅ SHAP értékek elmentve")
    except ImportError:
        print("  (SHAP: pip install shap)")


# ════════════════════════════════════════════════════════════
# LECKE 05 – KRIPTOVALUTA BOT
# ════════════════════════════════════════════════════════════

def kripto_bot_esettanulmany() -> None:
    """Kriptovaluta árfolyam elemzés és egyszerű stratégia.

    ⚠️  FONTOS FIGYELMEZTETÉS:
      Ez oktatási célú szimulációs projekt!
      Valós kereskedéshez komoly kockázatmenedzsment szükséges.
      A múltbeli teljesítmény NEM garantálja a jövőbelit.

    TARTALOM:
      1. Szintetikus árfolyam-adatok generálása
      2. Technikai indikátorok (MA, RSI, Bollinger)
      3. Egyszerű mozgóátlag stratégia backtesting-je
      4. Teljesítménymetrikák (Sharpe-ráta, max drawdown)
    """
    print("\n" + "=" * 60)
    print("  LECKE 05 – Kripto Bot Esettanulmány")
    print("=" * 60)

    # ── Szintetikus BTC/USD adatok ───────────────────────────
    rng = np.random.default_rng(42)
    n_nap = 365 * 2   # 2 év

    # Geometric Brownian Motion (pénzügyi árszimuláció)
    mu = 0.001        # napi drift
    sigma = 0.03      # napi volatilitás
    S0 = 30_000       # kezdőár (USD)

    returns = rng.normal(mu, sigma, n_nap)
    arak = S0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "datum": pd.date_range("2022-01-01", periods=n_nap, freq="D"),
        "ar":    arak,
        "volumen": rng.integers(1_000_000, 50_000_000, n_nap),
    })

    # ── Technikai indikátorok ────────────────────────────────
    def moving_average(serie: pd.Series, ablak: int) -> pd.Series:
        """Egyszerű mozgóátlag."""
        return serie.rolling(window=ablak).mean()

    def rsi(serie: pd.Series, ablak: int = 14) -> pd.Series:
        """RSI (Relative Strength Index): 0-100, >70 = túlvett, <30 = túladott."""
        delta = serie.diff()
        gain = delta.where(delta > 0, 0).rolling(window=ablak).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=ablak).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def bollinger_bands(serie: pd.Series, ablak: int = 20, std_szorzo: float = 2.0):
        """Bollinger Bands: közép ± 2 szórás."""
        kozep = serie.rolling(ablak).mean()
        std   = serie.rolling(ablak).std()
        return kozep + std_szorzo * std, kozep, kozep - std_szorzo * std

    df["ma_20"]  = moving_average(df["ar"], 20)
    df["ma_50"]  = moving_average(df["ar"], 50)
    df["rsi_14"] = rsi(df["ar"])
    df["bb_fels"], df["bb_kozep"], df["bb_also"] = bollinger_bands(df["ar"])

    # ── Egyszerű mozgóátlag crossover stratégia ──────────────
    # Vétel: MA20 keresztezi MA50-et felfelé
    # Eladás: MA20 keresztezi MA50-et lefelé
    df = df.dropna()
    df["signal"] = np.where(df["ma_20"] > df["ma_50"], 1, -1)
    df["signal_valtozas"] = df["signal"].diff()
    df["vetel"]  = df["signal_valtozas"] > 0
    df["eladas"] = df["signal_valtozas"] < 0

    # ── Backtesting ──────────────────────────────────────────
    tokeal = 100_000
    portfolio = tokeal
    pozicio = 0
    portfolio_history = []

    for _, row in df.iterrows():
        if row["vetel"] and portfolio > 0:
            pozicio = portfolio / row["ar"]
            portfolio = 0

        elif row["eladas"] and pozicio > 0:
            portfolio = pozicio * row["ar"]
            pozicio = 0

        portfolio_history.append(portfolio + pozicio * row["ar"])

    df["portfolio_ertek"] = portfolio_history

    # Teljesítménymetrikák
    vegso_ertek = portfolio_history[-1]
    hozam = (vegso_ertek - tokeal) / tokeal

    napi_hozamok = pd.Series(portfolio_history).pct_change().dropna()
    sharpe = napi_hozamok.mean() / napi_hozamok.std() * np.sqrt(252)

    portfoliо_max = pd.Series(portfolio_history).cummax()
    max_drawdown = ((pd.Series(portfolio_history) - portfoliо_max) / portfoliо_max).min()

    print(f"\nBacktest eredmények (2 év):")
    print(f"  Buy & Hold hozam: {(arak[-1]/arak[0]-1)*100:.1f}%")
    print(f"  Stratégia hozam:  {hozam*100:.1f}%")
    print(f"  Sharpe-ráta:      {sharpe:.3f}")
    print(f"  Max Drawdown:     {max_drawdown*100:.1f}%")

    # ── Vizualizáció ─────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Ár + MA + Bollinger + jelek
    axes[0].plot(df["datum"], df["ar"], color="#2C3E50", lw=1, alpha=0.8, label="BTC/USD")
    axes[0].plot(df["datum"], df["ma_20"], color="#3498DB", lw=1.5, label="MA(20)")
    axes[0].plot(df["datum"], df["ma_50"], color="#E74C3C", lw=1.5, label="MA(50)")
    axes[0].fill_between(df["datum"], df["bb_also"], df["bb_fels"], alpha=0.1, color="gray")
    axes[0].scatter(df[df["vetel"]]["datum"],  df[df["vetel"]]["ar"],  marker="^", color="green", s=60)
    axes[0].scatter(df[df["eladas"]]["datum"], df[df["eladas"]]["ar"], marker="v", color="red",   s=60)
    axes[0].set_ylabel("Ár (USD)")
    axes[0].legend(loc="upper left")
    axes[0].set_title("BTC/USD + Technikai indikátorok + Kereskedési jelek")

    # RSI
    axes[1].plot(df["datum"], df["rsi_14"], color="#9B59B6", lw=1)
    axes[1].axhline(70, color="red",   lw=1, linestyle="--", alpha=0.5)
    axes[1].axhline(30, color="green", lw=1, linestyle="--", alpha=0.5)
    axes[1].fill_between(df["datum"], 30, 70, alpha=0.05, color="gray")
    axes[1].set_ylabel("RSI(14)")
    axes[1].set_ylim(0, 100)

    # Portfolio érték
    axes[2].plot(df["datum"], df["portfolio_ertek"] / tokeal * 100 - 100,
                 color="#27AE60", lw=2, label="Stratégia")
    hold = (df["ar"] / df["ar"].iloc[0] - 1) * 100
    axes[2].plot(df["datum"], hold, color="#E74C3C", lw=1, alpha=0.5, label="Buy & Hold")
    axes[2].axhline(0, color="gray", lw=0.5)
    axes[2].set_ylabel("Kumulált hozam (%)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_kripto_bot.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# LECKE 06 – HÁLÓZATELEMZÉS
# ════════════════════════════════════════════════════════════

def halozatelemzes_esettanulmany() -> None:
    """Random hálózat elemzés NetworkX-szel.

    FELHASZNÁLÁSI TERÜLETEK:
      - Közösségi háló elemzés (influencerek)
      - Ellátási lánc kockázatelemzés
      - Pénzügyi tranzakciós hálózatok (fraud detekció)
      - IT infrastruktúra gráfelemzés

    METRIKÁK:
      Degree centrality: Közvetlen kapcsolatok száma
      Betweenness: Közvetítő szerep a hálózatban
      Closeness: Átlagos távolság más csúcsoktól
      PageRank: Globális fontosság (Google algoritmusa)
    """
    print("\n" + "=" * 60)
    print("  LECKE 06 – Hálózatelemzés")
    print("=" * 60)

    try:
        import networkx as nx
        from networkx.algorithms import community
    except ImportError:
        print("  NetworkX nem telepítve: pip install networkx")
        return

    # ── Barabási–Albert (scale-free) hálózat generálása ─────
    # Valós közösségi hálók hasonló struktúrát követnek
    G = nx.barabasi_albert_graph(n=100, m=2, seed=42)

    # ── Centralitás metrikák ──────────────────────────────────
    degree_cent     = nx.degree_centrality(G)
    between_cent    = nx.betweenness_centrality(G)
    closeness_cent  = nx.closeness_centrality(G)
    pagerank        = nx.pagerank(G, alpha=0.85)

    metrics_df = pd.DataFrame({
        "csucs":        list(G.nodes()),
        "degree":       [G.degree(n) for n in G.nodes()],
        "degree_cent":  [degree_cent[n] for n in G.nodes()],
        "between_cent": [between_cent[n] for n in G.nodes()],
        "pagerank":     [pagerank[n] for n in G.nodes()],
    }).sort_values("pagerank", ascending=False)

    print(f"\nHálózat alapadatok:")
    print(f"  Csúcsok:         {G.number_of_nodes()}")
    print(f"  Élek:            {G.number_of_edges()}")
    print(f"  Átlag fokszám:   {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  Klaszterezettség:{nx.average_clustering(G):.4f}")
    print(f"  Átm. úthossz:    {nx.average_shortest_path_length(G):.3f}")

    print(f"\nTop 5 legfontosabb csúcs (PageRank):")
    print(metrics_df.head(5)[["csucs", "degree", "pagerank", "between_cent"]].to_string(index=False))

    # ── Közösségdetektálás (Louvain) ─────────────────────────
    try:
        kozossegek = list(community.greedy_modularity_communities(G))
        print(f"\n  Detektált közösségek száma: {len(kozossegek)}")
        for i, k in enumerate(kozossegek[:3]):
            print(f"    Közösség {i+1}: {len(k)} csúcs")
    except Exception:
        kozossegek = []

    # ── Fokszám-eloszlás (Power Law) ─────────────────────────
    fokok = [d for _, d in G.degree()]
    fokszam_count = pd.Series(fokok).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Hálózat megjelenítése
    pos = nx.spring_layout(G, seed=42, k=0.3)
    node_sizes = [pagerank[n] * 5000 for n in G.nodes()]
    node_colors = [between_cent[n] for n in G.nodes()]

    nx.draw_networkx(
        G, pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cm.plasma,
        edge_color="#BDC3C7",
        alpha=0.9,
        ax=axes[0],
    )
    sm = plt.cm.ScalarMappable(cmap=cm.plasma,
                                 norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    plt.colorbar(sm, ax=axes[0], label="Betweenness centrality")
    axes[0].set_title("Hálózat (csúcs mérete = PageRank)")
    axes[0].axis("off")

    # Fokszám-eloszlás (Power Law log-log skálán)
    axes[1].loglog(fokszam_count.index, fokszam_count.values, "o", color="#3498DB", alpha=0.7)
    axes[1].set_xlabel("Fokszám (log)")
    axes[1].set_ylabel("Csúcsok száma (log)")
    axes[1].set_title("Fokszám-eloszlás (Power Law = scale-free hálózat)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_halozat.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    ai_onboarding_esettanulmany()
    kripto_bot_esettanulmany()
    halozatelemzes_esettanulmany()

    print("\n✅ Leckék 04/05/06 sikeresen lefutottak!")
    print("🎓 Gratulálunk, a kurzus minden leckéje elvégezve!")