"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 18 – Plotly Express – Interaktív Vizualizáció         ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - Scatter, line, bar, histogram, box, violin
  - Heatmap és korreláció
  - 3D scatter, animált plot
  - Facet grid (többpaneles ábra)
  - Dashboard: Dash / HTML mentés

PLOTLY vs. MATPLOTLIB:
  Matplotlib: statikus, tudományos publikációkba
  Plotly:     interaktív, dashboardokhoz, prezentációkhoz
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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
# ADATHALMAZ
# ════════════════════════════════════════════════════════════

def adat_generalas() -> pd.DataFrame:
    """Szintetikus e-commerce adathalmaz vizualizációhoz."""
    rng = np.random.default_rng(42)
    n = 500

    return pd.DataFrame({
        "datum":       pd.date_range("2023-01-01", periods=n, freq="D"),
        "bevetel":     rng.integers(50_000, 500_000, n).astype(float),
        "tranzakcio":  rng.integers(10, 200, n),
        "kategoria":   rng.choice(["Elektronika", "Ruházat", "Élelmiszer", "Sport"], n),
        "varos":       rng.choice(["Budapest", "Debrecen", "Pécs", "Győr", "Miskolc"], n),
        "ugyfel_kor":  rng.integers(18, 70, n),
        "megelégedettség": rng.uniform(1, 5, n).round(1),
        "visszater":   rng.choice([True, False], n, p=[0.6, 0.4]),
    })


# ════════════════════════════════════════════════════════════
# 1. ALAP DIAGRAMOK
# ════════════════════════════════════════════════════════════

def alap_diagramok(df: pd.DataFrame) -> None:
    """Scatter, bar, histogram, box alapdiagramok."""

    # ── Scatter ─────────────────────────────────────────────
    fig = px.scatter(
        df,
        x="tranzakcio",
        y="bevetel",
        color="kategoria",
        size="ugyfel_kor",
        hover_data=["varos", "datum"],
        title="Tranzakciók vs. Bevétel (kategóriánként)",
        labels={"tranzakcio": "Tranzakciók száma", "bevetel": "Bevétel (Ft)"},
        template="plotly_white",
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color="white")))
    fig.write_html(OUTPUT_DIR / "18_scatter.html")
    print("  📊 Scatter mentve: outputs/18_scatter.html")

    # ── Bar – havi összesítő ─────────────────────────────────
    havi = df.copy()
    havi["honap"] = havi["datum"].dt.to_period("M").astype(str)
    havi_agg = havi.groupby(["honap", "kategoria"])["bevetel"].sum().reset_index()

    fig = px.bar(
        havi_agg,
        x="honap",
        y="bevetel",
        color="kategoria",
        barmode="stack",
        title="Havi bevétel kategóriánként",
        labels={"bevetel": "Bevétel (Ft)", "honap": "Hónap"},
        template="plotly_white",
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_html(OUTPUT_DIR / "18_bar_stack.html")
    print("  📊 Bar (stacked) mentve: outputs/18_bar_stack.html")

    # ── Box Plot – eloszlás kategóriánként ──────────────────
    fig = px.box(
        df,
        x="kategoria",
        y="bevetel",
        color="kategoria",
        points="outliers",
        title="Bevétel eloszlása kategóriánként (Box Plot)",
        template="plotly_white",
        notched=True,   # 95%-os CI a mediánra
    )
    fig.write_html(OUTPUT_DIR / "18_box.html")
    print("  📊 Box plot mentve: outputs/18_box.html")


# ════════════════════════════════════════════════════════════
# 2. IDŐSORok
# ════════════════════════════════════════════════════════════

def idosor_vizualizacio(df: pd.DataFrame) -> None:
    """Idősor vonaldiagram mozgóátlaggal."""

    napi = df.groupby("datum")["bevetel"].sum().reset_index()
    napi["mozgo_atlag_7"] = napi["bevetel"].rolling(window=7, center=True).mean()
    napi["mozgo_atlag_30"] = napi["bevetel"].rolling(window=30, center=True).mean()

    fig = go.Figure()

    # Napi adat (halvány)
    fig.add_trace(go.Scatter(
        x=napi["datum"], y=napi["bevetel"],
        name="Napi bevétel",
        line=dict(color="#BDC3C7", width=1),
        opacity=0.5,
    ))

    # 7 napos mozgóátlag
    fig.add_trace(go.Scatter(
        x=napi["datum"], y=napi["mozgo_atlag_7"],
        name="7 napos átlag",
        line=dict(color="#3498DB", width=2),
    ))

    # 30 napos mozgóátlag
    fig.add_trace(go.Scatter(
        x=napi["datum"], y=napi["mozgo_atlag_30"],
        name="30 napos átlag",
        line=dict(color="#E74C3C", width=2.5),
    ))

    fig.update_layout(
        title="Napi bevétel és mozgóátlagok",
        xaxis_title="Dátum",
        yaxis_title="Bevétel (Ft)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    fig.write_html(OUTPUT_DIR / "18_idosor.html")
    print("  📊 Idősor mentve: outputs/18_idosor.html")


# ════════════════════════════════════════════════════════════
# 3. KORRELÁCIÓ HŐTÉRKÉP
# ════════════════════════════════════════════════════════════

def korrelacio_heatmap(df: pd.DataFrame) -> None:
    """Interaktív korreláció mátrix hőtérkép.

    Plotly heatmap Matplotlib helyett:
      + Hover: pontos értékek
      + Kattintásra ki-be kapcsolható
    """
    num_cols = ["bevetel", "tranzakcio", "ugyfel_kor", "megelégedettség"]
    korr = df[num_cols].corr().round(3)

    fig = px.imshow(
        korr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Korreláció Mátrix",
        template="plotly_white",
    )
    fig.update_layout(width=600, height=500)
    fig.write_html(OUTPUT_DIR / "18_korrelacio_heatmap.html")
    print("  📊 Heatmap mentve: outputs/18_korrelacio_heatmap.html")


# ════════════════════════════════════════════════════════════
# 4. FACET GRID (TÖBBPANELES)
# ════════════════════════════════════════════════════════════

def facet_vizualizacio(df: pd.DataFrame) -> None:
    """Facet grid: minden kategória külön panelen."""

    fig = px.histogram(
        df,
        x="bevetel",
        facet_col="kategoria",
        facet_col_wrap=2,
        color="visszater",
        barmode="overlay",
        opacity=0.7,
        title="Bevétel eloszlása kategóriánként (visszatérő vs. új vásárló)",
        template="plotly_white",
        labels={"visszater": "Visszatérő vásárló"},
        nbins=30,
    )
    fig.write_html(OUTPUT_DIR / "18_facet_histogram.html")
    print("  📊 Facet histogram mentve: outputs/18_facet_histogram.html")


# ════════════════════════════════════════════════════════════
# 5. ANIMÁLT SCATTER
# ════════════════════════════════════════════════════════════

def animalt_scatter(df: pd.DataFrame) -> None:
    """Animált scatter: hónaponkénti fejlődés vizualizálása."""

    anim_df = df.copy()
    anim_df["honap"] = anim_df["datum"].dt.to_period("M").astype(str)

    havi_kat = (
        anim_df.groupby(["honap", "kategoria"])
        .agg(bevetel=("bevetel", "sum"), tranzakcio=("tranzakcio", "sum"))
        .reset_index()
    )

    fig = px.scatter(
        havi_kat,
        x="tranzakcio",
        y="bevetel",
        color="kategoria",
        size="bevetel",
        animation_frame="honap",
        title="Havi fejlődés animálva",
        template="plotly_white",
        range_x=[0, havi_kat["tranzakcio"].max() * 1.1],
        range_y=[0, havi_kat["bevetel"].max() * 1.1],
    )
    fig.write_html(OUTPUT_DIR / "18_animalt_scatter.html")
    print("  📊 Animált scatter mentve: outputs/18_animalt_scatter.html")


# ════════════════════════════════════════════════════════════
# 6. DASHBOARD – KOMBINÁLT DASHBOARD
# ════════════════════════════════════════════════════════════

def dashboard_html(df: pd.DataFrame) -> None:
    """Multi-panel dashboard egyetlen HTML fájlba."""

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Bevétel eloszlása",
            "Kategóriánkénti átlagbevétel",
            "Megelégedettség vs. Bevétel",
            "Visszatérő vásárlók aránya",
        ],
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"},   {"type": "pie"}]],
    )

    # Panel 1: Histogram
    fig.add_trace(go.Histogram(x=df["bevetel"], nbinsx=40, name="Bevétel",
                               marker_color="#3498DB", opacity=0.7), row=1, col=1)

    # Panel 2: Bar
    kat_mean = df.groupby("kategoria")["bevetel"].mean().sort_values(ascending=True)
    fig.add_trace(go.Bar(x=kat_mean.values, y=kat_mean.index, orientation="h",
                         name="Átlagbevétel", marker_color="#2ECC71"), row=1, col=2)

    # Panel 3: Scatter
    fig.add_trace(go.Scatter(x=df["megelégedettség"], y=df["bevetel"],
                              mode="markers", marker=dict(opacity=0.4, color="#9B59B6"),
                              name="Elégedettség-Bevétel"), row=2, col=1)

    # Panel 4: Pie
    visszater_count = df["visszater"].value_counts()
    fig.add_trace(go.Pie(labels=["Visszatérő", "Új"],
                          values=visszater_count.values,
                          hole=0.4,   # donut
                          marker_colors=["#2ECC71", "#E74C3C"]), row=2, col=2)

    fig.update_layout(
        height=700,
        title_text="📊 E-Commerce Elemzési Dashboard",
        showlegend=False,
        template="plotly_white",
    )
    fig.write_html(OUTPUT_DIR / "18_dashboard.html")
    print("  📊 Dashboard mentve: outputs/18_dashboard.html")


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LECKE 18 – Plotly Express Vizualizáció")
    print("=" * 55)

    df = adat_generalas()
    print(f"Adathalmaz: {df.shape}\n")

    alap_diagramok(df)
    idosor_vizualizacio(df)
    korrelacio_heatmap(df)
    facet_vizualizacio(df)
    animalt_scatter(df)
    dashboard_html(df)

    print("\n✅ Lecke 18 sikeresen lefutott!")
    print("📁 Minden interaktív grafikon: outputs/ mappában")
    print("➡️  Következő: 20_stat_korrelacio.py")