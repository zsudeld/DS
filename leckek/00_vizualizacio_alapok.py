"""
00 – Vizualizáció alapok: Matplotlib & Seaborn
===============================================
Célok:
  - Matplotlib Figure/Axes felépítés megértése
  - Alap diagramtípusok (vonal, oszlop, kördiagram, hisztogram, scatter)
  - Seaborn statisztikai vizualizációk (boxplot, heatmap, pairplot)
  - Professzionális stílus, tengelyfeliratok, mentés PNG-be

Mikor használd Matplotlib/Seaborn helyett Plotly-t?
  → Matplotlib/Seaborn: statikus riportok, publikációk, egyszerű elemzés
  → Plotly: interaktív dashboard, web/HTML export (lásd: 18_plotly_express.py)
"""

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
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


# ── Kimeneti mappa ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Stílus beállítása ─────────────────────────────────────────────────────────
# Elérhető stílusok: 'seaborn-v0_8', 'ggplot', 'bmh', 'fivethirtyeight'
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = sns.color_palette("husl", 8)   # 8 jól elkülönülő szín


# ─────────────────────────────────────────────────────────────────────────────
# 1. MATPLOTLIB ALAPOK – Figure és Axes objektumok
# ─────────────────────────────────────────────────────────────────────────────

def matplotlib_alapok():
    """
    A Matplotlib két fő objektuma:
      - Figure: az egész „vászon" (pl. A4-es lap)
      - Axes:   egy adott diagram-terület a vásznon belül
    Egy Figure-ön belül több Axes is lehet (subplots).
    """
    print("\n=== 1. Matplotlib alapok ===")

    # Szintetikus adat
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # fig, ax = plt.subplots() → 1 db diagram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 sor, 2 oszlop

    # --- Bal panel: vonaldiagram ---
    axes[0].plot(x, y1, label="sin(x)", color=COLORS[0], linewidth=2)
    axes[0].plot(x, y2, label="cos(x)", color=COLORS[1], linewidth=2, linestyle="--")
    axes[0].set_title("Trigonometrikus függvények", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("x (radián)")
    axes[0].set_ylabel("y érték")
    axes[0].legend()
    axes[0].set_xlim(0, 2 * np.pi)

    # --- Jobb panel: kitöltött terület ---
    axes[1].fill_between(x, y1, y2, alpha=0.3, color=COLORS[2], label="Különbség")
    axes[1].plot(x, y1, color=COLORS[0])
    axes[1].plot(x, y2, color=COLORS[1])
    axes[1].set_title("Kitöltött terület (fill_between)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("x (radián)")
    axes[1].legend()

    plt.tight_layout()  # Automatikus margó igazítás
    fpath = OUTPUT_DIR / "00a_matplotlib_alapok.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mentve: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. ALAP DIAGRAMTÍPUSOK
# ─────────────────────────────────────────────────────────────────────────────

def alap_diagramtipusok():
    """
    A leggyakoribb diagram típusok és mikor érdemes használni őket:
      - Vonaldiagram:  időbeli változás, folytonos adat
      - Oszlopdiagram: kategóriák összehasonlítása
      - Hisztogram:    eloszlás vizualizálása
      - Scatter plot:  két változó kapcsolata
      - Kördiagram:    arányok (max 5-6 szelet, különben olvashatatlan!)
    """
    print("\n=== 2. Alap diagramtípusok ===")

    np.random.seed(42)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- Vonaldiagram: havi bevétel ---
    ax1 = fig.add_subplot(gs[0, 0])
    honapok = ["Jan", "Feb", "Már", "Ápr", "Máj", "Jún",
               "Júl", "Aug", "Sze", "Okt", "Nov", "Dec"]
    bevetel = [120, 132, 145, 138, 160, 175, 168, 155, 170, 182, 195, 210]
    ax1.plot(honapok, bevetel, marker="o", color=COLORS[0], linewidth=2, markersize=6)
    ax1.fill_between(range(12), bevetel, alpha=0.15, color=COLORS[0])
    ax1.set_title("Havi bevétel (M Ft)", fontweight="bold")
    ax1.set_ylabel("Bevétel (M Ft)")
    ax1.tick_params(axis="x", rotation=45)

    # --- Oszlopdiagram: termékek értékesítése ---
    ax2 = fig.add_subplot(gs[0, 1])
    termekek = ["Termék A", "Termék B", "Termék C", "Termék D", "Termék E"]
    ertekesites = [340, 215, 480, 125, 290]
    bars = ax2.bar(termekek, ertekesites, color=COLORS[:5], edgecolor="white", linewidth=0.5)
    # Értékek megjelenítése az oszlopok tetején
    for bar, val in zip(bars, ertekesites):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{val}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_title("Termékek értékesítése (db)", fontweight="bold")
    ax2.set_ylabel("Darabszám")
    ax2.tick_params(axis="x", rotation=20)

    # --- Hisztogram: jövedelemeloszlás ---
    ax3 = fig.add_subplot(gs[0, 2])
    jovedelmek = np.random.lognormal(mean=10.5, sigma=0.5, size=1000)  # Jobbra torzított
    ax3.hist(jovedelmek, bins=40, color=COLORS[2], edgecolor="white", linewidth=0.3, alpha=0.85)
    ax3.axvline(np.median(jovedelmek), color="red", linestyle="--", linewidth=1.5, label=f"Medián: {np.median(jovedelmek):.0f}")
    ax3.axvline(np.mean(jovedelmek), color="orange", linestyle="--", linewidth=1.5, label=f"Átlag: {np.mean(jovedelmek):.0f}")
    ax3.set_title("Jövedelemeloszlás", fontweight="bold")
    ax3.set_xlabel("Jövedelem (Ft)")
    ax3.set_ylabel("Gyakoriság")
    ax3.legend(fontsize=8)

    # --- Scatter plot: magasság vs. súly ---
    ax4 = fig.add_subplot(gs[1, 0])
    magassag = np.random.normal(170, 10, 200)
    suly = 0.5 * magassag - 40 + np.random.normal(0, 8, 200)
    nemek = np.random.choice(["Férfi", "Nő"], 200, p=[0.5, 0.5])
    for nem, szin in zip(["Férfi", "Nő"], [COLORS[0], COLORS[3]]):
        mask = nemek == nem
        ax4.scatter(magassag[mask], suly[mask], color=szin, alpha=0.6, label=nem, s=25)
    ax4.set_title("Magasság vs. Súly", fontweight="bold")
    ax4.set_xlabel("Magasság (cm)")
    ax4.set_ylabel("Súly (kg)")
    ax4.legend()

    # --- Kördiagram: piaci részesedés ---
    ax5 = fig.add_subplot(gs[1, 1])
    cegek = ["Vállalat A", "Vállalat B", "Vállalat C", "Vállalat D", "Egyéb"]
    reszesedes = [35, 25, 20, 12, 8]
    explode = (0.05, 0, 0, 0, 0)  # Az első szelet kiemelése
    wedges, texts, autotexts = ax5.pie(
        reszesedes, labels=cegek, autopct="%1.1f%%",
        colors=COLORS[:5], explode=explode, startangle=90
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax5.set_title("Piaci részesedés", fontweight="bold")

    # --- Vízszintes oszlopdiagram: TOP termékek ---
    ax6 = fig.add_subplot(gs[1, 2])
    kategoriak = ["Sport", "Elektronika", "Ruházat", "Élelmiszer", "Bútor"]
    visszakuldesi_arany = [8.2, 12.5, 15.1, 3.3, 6.7]
    bars_h = ax6.barh(kategoriak, visszakuldesi_arany, color=COLORS[4:9], edgecolor="white")
    ax6.set_title("Visszaküldési arány (%)", fontweight="bold")
    ax6.set_xlabel("Visszaküldés (%)")
    ax6.axvline(np.mean(visszakuldesi_arany), color="red", linestyle="--",
                linewidth=1, label=f"Átlag: {np.mean(visszakuldesi_arany):.1f}%")
    ax6.legend(fontsize=8)

    fig.suptitle("Alap diagramtípusok – Matplotlib", fontsize=15, fontweight="bold", y=1.01)

    fpath = OUTPUT_DIR / "00b_alap_diagramtipusok.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mentve: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEABORN STATISZTIKAI VIZUALIZÁCIÓK
# ─────────────────────────────────────────────────────────────────────────────

def seaborn_statisztikai():
    """
    Seaborn = Matplotlib tetejére épített, statisztikai fókuszú könyvtár.
    Előnyök:
      - Automatikus konfidencia-intervallumok
      - Könnyen kezeli a pandas DataFrame-eket
      - Szép alapértelmezett stílusok
    """
    print("\n=== 3. Seaborn statisztikai vizualizációk ===")

    np.random.seed(42)

    # Szintetikus HR adatbázis
    n = 300
    df = pd.DataFrame({
        "Kor": np.random.randint(22, 60, n),
        "Fizetes": np.random.normal(500_000, 150_000, n).clip(200_000, 1_200_000),
        "Osztaly": np.random.choice(["IT", "Sales", "HR", "Pénzügy", "Marketing"], n),
        "Teljesitmeny": np.random.choice(["Gyenge", "Átlagos", "Jó", "Kiváló"], n,
                                          p=[0.1, 0.3, 0.4, 0.2]),
        "Tapasztalat_ev": np.random.randint(0, 25, n),
        "Elegedettseg": np.random.uniform(1, 10, n)
    })
    # Összefüggés: több tapasztalat → magasabb fizetés (zajjal)
    df["Fizetes"] += df["Tapasztalat_ev"] * 15_000

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Boxplot: fizetés osztályonként ---
    sns.boxplot(data=df, x="Osztaly", y="Fizetes", ax=axes[0, 0],
                palette="husl", order=["IT", "Sales", "HR", "Pénzügy", "Marketing"])
    axes[0, 0].set_title("Fizetés osztályonként (Boxplot)", fontweight="bold")
    axes[0, 0].set_xlabel("")
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0, 0].tick_params(axis="x", rotation=20)

    # --- Violin plot: pontosabb eloszlás ---
    sns.violinplot(data=df, x="Teljesitmeny", y="Fizetes", ax=axes[0, 1],
                   palette="muted",
                   order=["Gyenge", "Átlagos", "Jó", "Kiváló"],
                   inner="quartile")
    axes[0, 1].set_title("Fizetés teljesítmény szerint (Violin)", fontweight="bold")
    axes[0, 1].set_xlabel("")
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # --- Heatmap: osztályok × teljesítmény ---
    pivot = df.groupby(["Osztaly", "Teljesitmeny"]).size().unstack(fill_value=0)
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                ax=axes[0, 2], linewidths=0.5,
                cbar_kws={"label": "Alkalmazottak száma"})
    axes[0, 2].set_title("Alkalmazottak száma (Heatmap)", fontweight="bold")
    axes[0, 2].set_xlabel("Teljesítmény")
    axes[0, 2].set_ylabel("Osztály")

    # --- Scatter + regressziós vonal ---
    sns.regplot(data=df, x="Tapasztalat_ev", y="Fizetes", ax=axes[1, 0],
                scatter_kws={"alpha": 0.4, "s": 20, "color": COLORS[0]},
                line_kws={"color": "red", "linewidth": 2})
    axes[1, 0].set_title("Tapasztalat vs. Fizetés (regresszióval)", fontweight="bold")
    axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # --- Barplot: átlagos elégedettség osztályonként (CI-val) ---
    sns.barplot(data=df, x="Osztaly", y="Elegedettseg", ax=axes[1, 1],
                palette="husl", capsize=0.1,
                order=["IT", "Sales", "HR", "Pénzügy", "Marketing"])
    axes[1, 1].set_title("Átlagos elégedettség (95% CI)", fontweight="bold")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Elégedettség (1–10)")
    axes[1, 1].tick_params(axis="x", rotation=20)

    # --- KDE plot: korcsoportok eloszlása ---
    for osztaly in ["IT", "Sales", "HR"]:
        subset = df[df["Osztaly"] == osztaly]["Kor"]
        sns.kdeplot(subset, ax=axes[1, 2], label=osztaly, fill=True, alpha=0.3)
    axes[1, 2].set_title("Korcsoport-eloszlás (KDE)", fontweight="bold")
    axes[1, 2].set_xlabel("Kor")
    axes[1, 2].legend()

    fig.suptitle("Seaborn – Statisztikai vizualizációk", fontsize=15, fontweight="bold")
    plt.tight_layout()

    fpath = OUTPUT_DIR / "00c_seaborn_statisztikai.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mentve: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PAIRPLOT – Több változó kapcsolata egyszerre
# ─────────────────────────────────────────────────────────────────────────────

def pairplot_demo():
    """
    Pairplot: minden változópár scatter plotját egyszerre mutatja.
    Diagnálisban az eloszlást (KDE/hisztogram) láthatod.
    Nagyon hasznos az EDA (exploratív adatelemzés) első lépéseként.
    Figyelem: sok változónál lassú (max ~8 változóig érdemes használni).
    """
    print("\n=== 4. Pairplot ===")

    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "Ár": np.random.normal(5000, 1500, n).clip(1000),
        "Méret_m2": np.random.normal(75, 20, n).clip(20),
        "Szobák": np.random.randint(1, 6, n).astype(float),
        "Korév": np.random.randint(1, 50, n).astype(float),
        "Típus": np.random.choice(["Lakás", "Ház", "Tégla"], n)
    })
    df["Ár"] += df["Méret_m2"] * 50  # Összefüggés

    g = sns.pairplot(df, hue="Típus", palette="husl",
                     diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
    g.figure.suptitle("Pairplot – Ingatlan adatok", y=1.02, fontsize=14, fontweight="bold")

    fpath = OUTPUT_DIR / "00d_pairplot.png"
    g.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Mentve: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# FŐPROGRAM
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  00 – Vizualizáció alapok: Matplotlib & Seaborn")
    print("=" * 60)

    matplotlib_alapok()
    alap_diagramtipusok()
    seaborn_statisztikai()
    pairplot_demo()

    print("\n✅ Kész! Nézd meg az 'outputs/' mappát a mentett PNG fájlokért.")
    print("\nKövetkező lépés: 18_plotly_express.py – Interaktív vizualizációk")