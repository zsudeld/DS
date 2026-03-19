"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 20 – Korrelációelemzés                                ║
║  LECKE 21 – Statisztikai Lineáris Regresszió (OLS)          ║
║  LECKE 22 – Normalizáció és Normalitástesztek               ║
║  LECKE 23 – Szignifikanciateszt (t-teszt, ANOVA, Chi²)      ║
╚══════════════════════════════════════════════════════════════╝

STATISZTIKA ≠ ML:
  Statisztika: Hipotézis tesztelés, ok-okozat, interpretálhatóság
  ML: Előrejelzés, mintafelismerés, pontosság maximalizálás
  → DS szakértőnek mindkét eszközkészlet kell!
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import warnings; warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
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
# KÖZÖS ADATHALMAZ
# ════════════════════════════════════════════════════════════

def adat_generalas(n: int = 300) -> pd.DataFrame:
    """Munkavállalói adathalmaz statisztikai elemzéshez."""
    rng = np.random.default_rng(42)

    kor      = rng.integers(22, 60, n).astype(float)
    tapaszt  = (kor - 22 + rng.integers(-3, 5, n)).clip(0).astype(float)
    bevetel  = (200_000 + tapaszt * 15_000 + kor * 2_000
                + rng.normal(0, 50_000, n))
    vegzettseg = rng.choice(["alap", "közép", "felső"], n, p=[0.3, 0.4, 0.3])
    osztaly    = rng.choice(["IT", "Értékesítés", "HR", "Pénzügy"], n)
    elegedettseg = rng.integers(1, 6, n).astype(float)
    marad      = (bevetel * 0.4e-5 + tapaszt * 0.1 + rng.normal(0, 1, n) > 1).astype(int)

    return pd.DataFrame({
        "kor": kor, "tapasztalat": tapaszt, "bevetel": bevetel,
        "vegzettseg": vegzettseg, "osztaly": osztaly,
        "elegedettseg": elegedettseg, "marad": marad,
    })


# ════════════════════════════════════════════════════════════
# LECKE 20 – KORRELÁCIÓELEMZÉS
# ════════════════════════════════════════════════════════════

def korrelacio_elemzes(df: pd.DataFrame) -> None:
    """Pearson, Spearman korreláció és vizualizáció.

    Pearson:  Lineáris összefüggés, normális eloszlást feltételez
    Spearman: Rangkorreláció, nemlineáris összefüggésre is érzékeny
    Kendall:  Kis mintánál megbízhatóbb

    Értelmezés:
      |r| < 0.3:  gyenge összefüggés
      |r| 0.3-0.7: közepes
      |r| > 0.7:  erős
    """
    print("=" * 55)
    print("  LECKE 20 – Korrelációelemzés")
    print("=" * 55)

    num_cols = ["kor", "tapasztalat", "bevetel", "elegedettseg"]

    # Pearson és Spearman összehasonlítás
    pearson  = df[num_cols].corr(method="pearson").round(3)
    spearman = df[num_cols].corr(method="spearman").round(3)

    print("\nPearson korreláció:")
    print(pearson.to_string())
    print("\nSpearman korreláció:")
    print(spearman.to_string())

    # Pár-korrelációk p-értékekkel
    print("\nPáros korrelációk (Pearson, p-értékkel):")
    for c1 in num_cols:
        for c2 in num_cols:
            if c1 < c2:
                r, p = stats.pearsonr(df[c1], df[c2])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"  {c1} ↔ {c2}: r={r:.3f}, p={p:.4f} {sig}")

    # Hőtérkép vizualizáció
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (korr_df, cim) in zip(axes, [(pearson, "Pearson"), (spearman, "Spearman")]):
        mask = np.triu(np.ones_like(korr_df, dtype=bool), k=1)  # felső háromszög elrejtése
        sns.heatmap(
            korr_df, annot=True, fmt=".3f",
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            mask=mask, ax=ax, square=True, linewidths=0.5,
        )
        ax.set_title(f"{cim} korreláció")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "20_korrelacio_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Scatter mátrix – vizuális exploráció
    g = sns.pairplot(df[num_cols], diag_kind="kde", plot_kws={"alpha": 0.3})
    g.fig.suptitle("Scatter mátrix (páronkénti összefüggések)", y=1.01, fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "20_pairplot.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# LECKE 21 – STATISZTIKAI LINEÁRIS REGRESSZIÓ (OLS)
# ════════════════════════════════════════════════════════════

def stat_linearis_regresszio(df: pd.DataFrame) -> None:
    """OLS regresszió statsmodels-szel.

    KÜLÖNBSÉG sklearn LinearRegression-tól:
      sklearn: előrejelzés fókusz (nincs p-érték, CI)
      statsmodels: statisztikai inferencia (p-értékek, CI, tesztek)

    OLS FELTÉTELEK:
      1. Linearitás
      2. Hibák normalitása
      3. Homoszkedaszticitás (konstans szórásnégyzet)
      4. Multikollinearitás hiánya (VIF < 10)
    """
    print("\n" + "=" * 55)
    print("  LECKE 21 – Statisztikai Lineáris Regresszió")
    print("=" * 55)

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        print("  statsmodels nem telepítve: pip install statsmodels")
        return

    # Formula interface – olvasható, R-stílusú szintaxis
    model = smf.ols(
        formula="bevetel ~ kor + tapasztalat + C(vegzettseg) + elegedettseg",
        data=df,
    ).fit()

    print(model.summary())

    # Fontos metrikák kiemelése
    print(f"\n📊 Összefoglaló:")
    print(f"  R²:         {model.rsquared:.4f}")
    print(f"  Adj. R²:    {model.rsquared_adj:.4f}")
    print(f"  F-stat:     {model.fvalue:.2f} (p={model.f_pvalue:.4e})")
    print(f"  AIC:        {model.aic:.2f}")
    print(f"  BIC:        {model.bic:.2f}")

    # Feltételek ellenőrzése
    maradek = model.resid

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Maradékok vs. illesztett értékek (homoszkedaszticitás)
    axes[0, 0].scatter(model.fittedvalues, maradek, alpha=0.4)
    axes[0, 0].axhline(0, color="red", lw=1)
    axes[0, 0].set_title("Maradékok vs. Illesztett értékek")
    axes[0, 0].set_xlabel("Illesztett értékek")
    axes[0, 0].set_ylabel("Maradékok")

    # 2. Q-Q plot (normalitás)
    stats.probplot(maradek, plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot (normalitás ellenőrzése)")

    # 3. Maradékok hisztogramja
    axes[1, 0].hist(maradek, bins=30, color="#3498DB", alpha=0.7, density=True)
    x_range = np.linspace(maradek.min(), maradek.max(), 100)
    axes[1, 0].plot(x_range, stats.norm.pdf(x_range, maradek.mean(), maradek.std()),
                    "r-", lw=2, label="Normális")
    axes[1, 0].set_title("Maradékok eloszlása")
    axes[1, 0].legend()

    # 4. Koefficiens plot (konfidencia-intervallumokkal)
    konfig_df = pd.DataFrame({
        "koef": model.params,
        "also_ci": model.conf_int()[0],
        "felso_ci": model.conf_int()[1],
        "p": model.pvalues,
    }).dropna().iloc[1:]   # intercept kihagyva

    colors = ["#2ECC71" if p < 0.05 else "#BDC3C7" for p in konfig_df["p"]]
    axes[1, 1].barh(range(len(konfig_df)), konfig_df["koef"], color=colors, alpha=0.8)
    axes[1, 1].errorbar(
        konfig_df["koef"], range(len(konfig_df)),
        xerr=[konfig_df["koef"] - konfig_df["also_ci"], konfig_df["felso_ci"] - konfig_df["koef"]],
        fmt="none", color="black", capsize=3,
    )
    axes[1, 1].axvline(0, color="red", lw=1, linestyle="--")
    axes[1, 1].set_yticks(range(len(konfig_df)))
    axes[1, 1].set_yticklabels(konfig_df.index, fontsize=8)
    axes[1, 1].set_title("Koefficiensek (95% CI) – zöld: szignifikáns")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "21_ols_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# LECKE 22 – NORMALIZÁCIÓ ÉS NORMALITÁSTESZTEK
# ════════════════════════════════════════════════════════════

def normalizacio_demo(df: pd.DataFrame) -> None:
    """Normalitástesztek és transzformációk.

    TESZTEK:
      Shapiro-Wilk:   kis minta (n<2000), legérzékenyebb
      Kolmogorov-Smirnov: nagy minta
      D'Agostino-Pearson: ferdeség + csúcsosság alapján

    H₀: az adat normális eloszlású
    Ha p < 0.05 → elvetjük H₀ → nem normális
    """
    print("\n" + "=" * 55)
    print("  LECKE 22 – Normalitástesztek")
    print("=" * 55)

    col = "bevetel"
    adatok = df[col].values

    # Normalitástesztek
    teszt_eredmenyek = {}

    stat_sw, p_sw = stats.shapiro(adatok[:500])  # max 5000 elem
    teszt_eredmenyek["Shapiro-Wilk"] = (stat_sw, p_sw)

    stat_ks, p_ks = stats.kstest(adatok, "norm", args=(adatok.mean(), adatok.std()))
    teszt_eredmenyek["Kolmogorov-Smirnov"] = (stat_ks, p_ks)

    stat_dp, p_dp = stats.normaltest(adatok)  # D'Agostino & Pearson
    teszt_eredmenyek["D'Agostino-Pearson"] = (stat_dp, p_dp)

    print(f"\nNormalitástesztek ({col}):")
    for nev, (stat, p) in teszt_eredmenyek.items():
        dontés = "✅ Normális" if p >= 0.05 else "❌ Nem normális"
        print(f"  {nev:<25} stat={stat:.4f}, p={p:.4f} → {dontés}")

    # Ferdeség és csúcsosság
    ferdesseg = stats.skew(adatok)
    csucs = stats.kurtosis(adatok)
    print(f"\n  Ferdeség (skewness): {ferdesseg:.4f}  {'(jobbra ferde)' if ferdesseg > 0.5 else ''}")
    print(f"  Csúcsosság (kurtosis): {csucs:.4f}")

    # Transzformáció összehasonlítás
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (adat, cim) in zip(axes, [
        (adatok,            "Eredeti"),
        (np.log1p(adatok),  "Log transzformált"),
        (np.sqrt(adatok),   "Sqrt transzformált"),
    ]):
        ax.hist(adat, bins=40, alpha=0.7, color="#3498DB", density=True)
        # Normális illesztés
        mu, std = adat.mean(), adat.std()
        x = np.linspace(adat.min(), adat.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), "r-", lw=2)
        _, p = stats.normaltest(adat)
        ax.set_title(f"{cim}\n(p-érték: {p:.4f})")
        ax.set_xlabel(col)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "22_normalitas_transzform.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# LECKE 23 – SZIGNIFIKANCIATESZT
# ════════════════════════════════════════════════════════════

def szignifikancia_demo(df: pd.DataFrame) -> None:
    """t-teszt, ANOVA, Chi-négyzet teszt.

    SZIGNIFIKANCIASZINT (α = 0.05):
      Ha p < 0.05: Elvetjük H₀ (statisztikailag szignifikáns)
      Ha p ≥ 0.05: Nem elvetjük H₀ (nincs elég bizonyíték)

    ⚠️  FONTOS:
      p < 0.05 ≠ fontos / nagy hatás
      Mindig nézd az effect size-t is! (Cohen's d, eta²)
    """
    print("\n" + "=" * 55)
    print("  LECKE 23 – Szignifikanciateszt")
    print("=" * 55)

    # ── 1. Egymintos t-teszt ─────────────────────────────────
    # H₀: Az átlagbevétel egyenlő 350 000 Ft-tal
    t_stat, p_val = stats.ttest_1samp(df["bevetel"], popmean=350_000)
    print(f"\n1. Egymintas t-teszt (H₀: átlag = 350 000 Ft)")
    print(f"   t={t_stat:.3f}, p={p_val:.4f} → {'❌ Elvetjük H₀' if p_val < 0.05 else '✅ Nem vetjük el H₀'}")

    # ── 2. Kétmintas t-teszt ─────────────────────────────────
    # H₀: IT és Értékesítés bevétele azonos
    it_bev   = df[df["osztaly"] == "IT"]["bevetel"]
    ert_bev  = df[df["osztaly"] == "Értékesítés"]["bevetel"]

    t_stat2, p_val2 = stats.ttest_ind(it_bev, ert_bev, equal_var=False)  # Welch's t
    cohen_d = (it_bev.mean() - ert_bev.mean()) / np.sqrt(
        (it_bev.std()**2 + ert_bev.std()**2) / 2
    )
    print(f"\n2. Kétmintas t-teszt (IT vs. Értékesítés bevétel)")
    print(f"   IT átlag: {it_bev.mean():,.0f} Ft  |  Értékesítés: {ert_bev.mean():,.0f} Ft")
    print(f"   t={t_stat2:.3f}, p={p_val2:.4f} → {'❌ Szignifikáns különbség' if p_val2 < 0.05 else '✅ Nincs szignifikáns különbség'}")
    print(f"   Cohen's d: {cohen_d:.3f} ({'nagy' if abs(cohen_d) > 0.8 else 'közepes' if abs(cohen_d) > 0.5 else 'kis'} hatás)")

    # ── 3. Egyutas ANOVA ─────────────────────────────────────
    # H₀: Minden osztály bevétele azonos
    csoportok = [df[df["osztaly"] == o]["bevetel"] for o in df["osztaly"].unique()]
    f_stat, p_anova = stats.f_oneway(*csoportok)
    print(f"\n3. ANOVA (összes osztály bevétele)")
    print(f"   F={f_stat:.3f}, p={p_anova:.4f}")
    if p_anova < 0.05:
        print("   ❌ Szignifikáns különbség van legalább egy pár között")
        # Post-hoc (Tukey) – pingouin-nal vagy iteratív t-tesztekkel
        try:
            import pingouin as pg
            posthoc = pg.pairwise_tests(data=df, dv="bevetel", between="osztaly")
            print("\n   Post-hoc tesztek (Pingouin):")
            print(posthoc[["A", "B", "T", "p-unc", "p-corr"]].to_string(index=False))
        except ImportError:
            print("   (Post-hoc: pip install pingouin)")

    # ── 4. Chi-négyzet teszt ─────────────────────────────────
    # H₀: A végzettség és a maradás független egymástól
    kontingencia = pd.crosstab(df["vegzettseg"], df["marad"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(kontingencia)

    print(f"\n4. Chi-négyzet teszt (Végzettség vs. Maradás)")
    print(f"   Kontingencia tábla:\n{kontingencia}")
    print(f"   χ²={chi2:.3f}, df={dof}, p={p_chi:.4f}")
    cramers_v = np.sqrt(chi2 / (len(df) * (min(kontingencia.shape) - 1)))
    print(f"   Cramér's V (hatáserő): {cramers_v:.4f}")
    if p_chi < 0.05:
        print("   ❌ Szignifikáns összefüggés van végzettség és maradás között!")

    # Vizualizáció: osztályok bevétel eloszlása
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    df.boxplot(column="bevetel", by="osztaly", ax=axes[0])
    axes[0].set_title("Bevétel eloszlása osztályonként")
    axes[0].set_xlabel("Osztály")
    axes[0].set_ylabel("Bevétel (Ft)")

    # Kontingencia heatmap
    sns.heatmap(
        kontingencia,
        annot=True, fmt="d",
        cmap="YlOrRd",
        ax=axes[1],
    )
    axes[1].set_title("Végzettség × Maradás (Chi-négyzet)")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "23_szignifikancia.png", dpi=150, bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    df = adat_generalas(n=400)

    korrelacio_elemzes(df)
    stat_linearis_regresszio(df)
    normalizacio_demo(df)
    szignifikancia_demo(df)

    print("\n✅ Leckék 20/21/22/23 sikeresen lefutottak!")
    print("➡️  Következő: 03_ai_databiz.py")