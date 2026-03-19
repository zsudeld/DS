"""
27 – Deployment & MLOps: Modell élesbe helyezése
==================================================
Célok:
  - Modell mentése és betöltése (joblib / pickle)
  - FastAPI REST API endpoint építése (modell kiszolgálása)
  - Model versioning alapok
  - Docker Compose leírás (komment alapú oktatás)
  - Monitoring: drift detektálás alapfogalmak

Tipikus ML lifecycle:
  Fejlesztés (Jupyter) → Kísérlet (MLflow) → Élesítés (FastAPI+Docker) → Monitoring

MEGJEGYZÉS: Ez a fájl bemutatja a teljes deployment workflow-t.
  A FastAPI szerver ténylegesen futtatható: python 27_deployment_mlops.py --serve
"""

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import joblib
    import json
    import warnings
    import sys
    import os
    from pathlib import Path
    from datetime import datetime

    warnings.filterwarnings("ignore")
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
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODELL TANÍTÁSA ÉS MENTÉSE
# ─────────────────────────────────────────────────────────────────────────────

def modell_tanitasa_es_mentese() -> dict:
    """
    Valós workflow: betanítjuk a modellt, majd mentjük a teljes Pipeline-t.

    Miért Pipeline-t mentünk, nem csak a modellt?
      Mert a Pipeline magában foglalja az előfeldolgozást is (scaler, encoder).
      Ha csak a modellt mentjük → production-ban hiányozni fog a scaling!
      → Mindig a teljes sklearn Pipeline-t mentsd.
    """
    print("\n=== 1. Modell betanítás és mentés ===")

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    # Szintetikus churn adathalmaz
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "kor":              np.random.randint(18, 70, n),
        "elofizetes_hetek": np.random.randint(1, 260, n),
        "havi_koltes":      np.random.exponential(5000, n).round(0),
        "bejelentkezesek":  np.random.randint(0, 50, n),
        "panaszok":         np.random.randint(0, 5, n),
    })
    # Churn logika
    churn_valoszinuseg = (
        0.3
        - 0.002 * df["elofizetes_hetek"]
        + 0.05  * df["panaszok"]
        - 0.003 * df["bejelentkezesek"]
    ).clip(0.05, 0.95)
    df["churn"] = (np.random.rand(n) < churn_valoszinuseg).astype(int)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: scaler + GBM
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("  Teszt kiértékelés:")
    print(classification_report(y_test, y_pred, target_names=["Marad", "Churn"]))

    # Modell mentése
    model_path = MODELS_DIR / "churn_pipeline_v1.pkl"
    joblib.dump(pipeline, model_path)
    print(f"  ✅ Modell mentve: {model_path}")

    # Metaadatok mentése (modell kártya)
    metadata = {
        "model_neve":    "churn_pipeline",
        "verzio":        "1.0.0",
        "betanitva":     datetime.now().isoformat(),
        "features":      list(X.columns),
        "target":        "churn",
        "algoritmus":    "GradientBoostingClassifier",
        "test_accuracy": float((y_pred == y_test).mean().round(4)),
        "sklearn_verzio": __import__("sklearn").__version__,
    }
    meta_path = MODELS_DIR / "churn_pipeline_v1_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Metaadatok mentve: {meta_path}")

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODELL BETÖLTÉSE ÉS PREDIKCIÓ
# ─────────────────────────────────────────────────────────────────────────────

def modell_betoltese_es_predikcios():
    """
    Modell betöltése és használata – ezt csinálja az API handler.
    Mindig ellenőrizd:
      - Megegyezik-e a feature lista a betanításkori listával?
      - Helyes-e a feature sorrendje?
      - Kompatibilis-e a scikit-learn verzió?
    """
    print("\n=== 2. Modell betöltés és predikció ===")

    model_path = MODELS_DIR / "churn_pipeline_v1.pkl"
    if not model_path.exists():
        print("  [!] Nincs mentett modell, futtasd először az 1. részt!")
        return

    pipeline = joblib.load(model_path)
    print(f"  ✅ Modell betöltve: {model_path}")

    # Tesztpredikciók
    uj_ugyfelek = pd.DataFrame({
        "kor":              [35, 55, 22],
        "elofizetes_hetek": [150, 10, 5],
        "havi_koltes":      [8000, 3000, 12000],
        "bejelentkezesek":  [30, 2, 45],
        "panaszok":         [0, 4, 1],
    })

    joslatok    = pipeline.predict(uj_ugyfelek)
    valoszinuseg = pipeline.predict_proba(uj_ugyfelek)[:, 1]

    print("\n  Predikciók:")
    for i, (joslatom, prob) in enumerate(zip(joslatok, valoszinuseg)):
        cimke = "🔴 CHURN" if joslatom == 1 else "🟢 Marad"
        print(f"  Ügyfél #{i+1}: {cimke}  (valószínűség: {prob:.1%})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. FASTAPI REST API KÓD (futtatható)
# ─────────────────────────────────────────────────────────────────────────────

FASTAPI_KOD = '''
"""
FastAPI REST API – Churn Prediction Service
Futtatás: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Teszt:    curl -X POST http://localhost:8000/predict \\
               -H "Content-Type: application/json" \\
               -d \'{"kor":35,"elofizetes_hetek":150,"havi_koltes":8000,"bejelentkezesek":30,"panaszok":0}\'
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── App és modell betöltés ───────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description="Ügyfélelőrejelzés – GBM pipeline",
)

# Singleton betöltés (egyszer tölt be, minden kérésnél ugyanazt használja)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(OUTPUT_DIR / "models/churn_pipeline_v1.pkl")
    return _pipeline

# ── Bemeneti séma (Pydantic) ─────────────────────────────────────────────────
class ChurnRequest(BaseModel):
    kor:              int   = Field(..., ge=18, le=100, description="Ügyfél kora")
    elofizetes_hetek: int   = Field(..., ge=0,  description="Előfizetés hossza (hetek)")
    havi_koltes:      float = Field(..., ge=0,  description="Havi átlagos költés (Ft)")
    bejelentkezesek:  int   = Field(..., ge=0,  description="Bejelentkezések száma (30 nap)")
    panaszok:         int   = Field(..., ge=0,  description="Panaszok száma (90 nap)")

class ChurnResponse(BaseModel):
    churn_valoszinuseg: float
    dontes:             str     # "marad" vagy "churn"
    kockazati_szint:    str     # "alacsony" / "közepes" / "magas"

# ── Endpointok ───────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "model_verzio": "1.0.0"}

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    try:
        pipeline = get_pipeline()
        X = pd.DataFrame([request.model_dump()])
        prob = float(pipeline.predict_proba(X)[0, 1])
        dontes = "churn" if prob >= 0.5 else "marad"
        if prob < 0.3:
            kockazat = "alacsony"
        elif prob < 0.6:
            kockazat = "közepes"
        else:
            kockazat = "magas"
        return ChurnResponse(
            churn_valoszinuseg=round(prob, 4),
            dontes=dontes,
            kockazati_szint=kockazat
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(ugyfelek: list[ChurnRequest]):
    pipeline = get_pipeline()
    X = pd.DataFrame([u.model_dump() for u in ugyfelek])
    probs = pipeline.predict_proba(X)[:, 1]
    return [{"churn_valoszinuseg": round(float(p), 4),
             "dontes": "churn" if p >= 0.5 else "marad"} for p in probs]
'''

def fastapi_kod_mentese():
    """Menti a FastAPI kódot egy futtatható fájlba."""
    api_path = OUTPUT_DIR / "churn_api.py"
    with open(api_path, "w", encoding="utf-8") as f:
        f.write(FASTAPI_KOD)
    print(f"\n=== 3. FastAPI kód mentve: {api_path} ===")
    print("  Futtatás: pip install fastapi uvicorn")
    print("            uvicorn churn_api:app --reload")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DOCKER LEÍRÁS (oktatás)
# ─────────────────────────────────────────────────────────────────────────────

DOCKERFILE = """# ── Dockerfile ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Függőségek (cache-elés rétegezéssel)
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn joblib scikit-learn pandas numpy

# Alkalmazás kód
COPY churn_api.py .
COPY outputs/models/ outputs/models/

EXPOSE 8000

CMD ["uvicorn", "churn_api:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKER_COMPOSE = """# ── docker-compose.yml ──────────────────────────────────────────────────────
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=outputs/models/churn_pipeline_v1.pkl
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped
"""

def docker_leiras_mentese():
    """Docker és docker-compose fájlok mentése."""
    (OUTPUT_DIR / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (OUTPUT_DIR / "docker-compose.yml").write_text(DOCKER_COMPOSE, encoding="utf-8")
    print(f"\n=== 4. Docker fájlok mentve: outputs/Dockerfile, docker-compose.yml ===")
    print("  Futtatás:")
    print("    docker build -t churn-api .")
    print("    docker run -p 8000:8000 churn-api")
    print("  Vagy docker-compose-szal:")
    print("    docker-compose up --build")


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA DRIFT DETEKTÁLÁS
# ─────────────────────────────────────────────────────────────────────────────

def drift_detektales():
    """
    Miért hal meg a modell production-ban?
      A valós adatok elkezdenek eltérni a tanítóadatoktól (data drift).
      Pl.: gazdasági változás → ügyfél viselkedés megváltozik
           szoftverfrissítés → egy feature értéke mindig 0 lesz

    Egyszerű drift detection:
      - Population Stability Index (PSI): > 0.25 → szignifikáns drift
      - KL-divergencia
      - Kolmogorov-Smirnov teszt (scipy.stats.ks_2samp)
      - Evidently AI könyvtár (teljes drift riport HTML-be)

    Mikor tanítsd újra?
      - PSI > 0.25 bármely jellemzőre
      - Predikciós pontosság > 5%-ot esett
      - Üzleti metrika romlik (churn arány nő predikció nélkül)
    """
    print("\n=== 5. Data Drift detektálás ===")

    np.random.seed(42)

    # Referencia eloszlás (tanítóadat)
    ref_bejelentkezesek = np.random.randint(0, 50, 1000)
    # Produkciós eloszlás (drift: kevesebb bejelentkezés → app probléma?)
    prod_bejelentkezesek = np.random.randint(0, 20, 1000)

    # PSI számítás
    def psi(ref, prod, n_bin=10):
        """Population Stability Index. Értelmezés: <0.1 stabil, 0.1-0.25 figyelj, >0.25 drift."""
        minv, maxv = min(ref.min(), prod.min()), max(ref.max(), prod.max())
        bins = np.linspace(minv, maxv, n_bin + 1)
        ref_arany  = np.histogram(ref,  bins=bins)[0] / len(ref)
        prod_arany = np.histogram(prod, bins=bins)[0] / len(prod)
        ref_arany  = np.where(ref_arany  == 0, 1e-6, ref_arany)
        prod_arany = np.where(prod_arany == 0, 1e-6, prod_arany)
        return np.sum((prod_arany - ref_arany) * np.log(prod_arany / ref_arany))

    psi_ertek = psi(ref_bejelentkezesek, prod_bejelentkezesek)
    print(f"\n  PSI (bejelentkezések): {psi_ertek:.3f}")
    if psi_ertek < 0.1:
        print("  ✅ Stabil eloszlás")
    elif psi_ertek < 0.25:
        print("  ⚠️  Kis mértékű drift – figyelj")
    else:
        print("  🔴 SZIGNIFIKÁNS DRIFT – újratanítás szükséges!")

    # KS teszt
    try:
        from scipy import stats
        ks_stat, ks_p = stats.ks_2samp(ref_bejelentkezesek, prod_bejelentkezesek)
        print(f"  KS teszt: statisztika={ks_stat:.3f}, p={ks_p:.2e}")
        print(f"  {'🔴 Szignifikáns drift (p < 0.05)' if ks_p < 0.05 else '✅ Nincs szignifikáns drift'}")
    except ImportError:
        print("  [!] scipy nem elérhető, KS teszt skip")

    # Drift vizualizáció
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ref_bejelentkezesek, bins=25, alpha=0.6, label="Referencia (tanítóadat)",
            color="#2196F3", density=True)
    ax.hist(prod_bejelentkezesek, bins=25, alpha=0.6, label="Produkció",
            color="#FF5722", density=True)
    ax.set_title(f"Data Drift – 'bejelentkezések' feature | PSI={psi_ertek:.3f}",
                 fontweight="bold")
    ax.set_xlabel("Bejelentkezések száma")
    ax.set_ylabel("Sűrűség")
    ax.legend()
    ax.text(0.97, 0.92, f"PSI = {psi_ertek:.3f}\n{'🔴 DRIFT!' if psi_ertek > 0.25 else '✅ OK'}",
            transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="#FFF3E0"))
    plt.tight_layout()
    fpath = OUTPUT_DIR / "27_drift_detektacias.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Mentve: {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. DEPLOYMENT CHECKLIST
# ─────────────────────────────────────────────────────────────────────────────

def deployment_checklist():
    print("""
=== 6. Deployment Checklist ===

  ✅ MODELL
     □ Pipeline-t mentettél (nem csak a modellt)
     □ Feature lista és sorrend dokumentálva
     □ Model card / metaadatok mentve (verzió, dátum, accuracy)
     □ Reprodukálhatóság: random seed rögzítve

  ✅ API
     □ Input validáció (Pydantic sémák)
     □ Hibaüzenet vs. stack trace (user ne lásson stack trace-t)
     □ /health endpoint (load balancer számára)
     □ Batch endpoint (ha >1 predikció kell)
     □ Authentikáció (API kulcs, JWT)

  ✅ INFRASTRUKTÚRA
     □ Docker image buildelhető
     □ Environment változók (nem hardkodolt útvonalak)
     □ Naplózás (logging, nem print)
     □ CI/CD pipeline (GitHub Actions / GitLab CI)

  ✅ MONITORING
     □ Predikciók naplózása (timestamp, input, output)
     □ PSI / KS drift check (hetente)
     □ Business KPI nyomon követése (pl. valós churn arány)
     □ Riaszt, ha a modell pontossága esik
    """)


# ─────────────────────────────────────────────────────────────────────────────
# FŐPROGRAM
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  27 – Deployment & MLOps")
    print("=" * 60)

    metadata = modell_tanitasa_es_mentese()
    modell_betoltese_es_predikcios()
    fastapi_kod_mentese()
    docker_leiras_mentese()
    drift_detektales()
    deployment_checklist()

    print("\n✅ Kész!")
    print("\nA teljes kurzust elvégezted! 🎉")
    print("Összefoglalás: README.md")