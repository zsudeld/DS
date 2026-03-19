"""
╔══════════════════════════════════════════════════════════════╗
║  LECKE 03 – AI + Üzleti Adatelemzés (LLM Integráció)        ║
╚══════════════════════════════════════════════════════════════╝

TANULÁSI CÉLOK:
  - LLM (Large Language Model) beillesztése az adatpipelineba
  - Szöveges adatok elemzése AI-jal (sentiment, összefoglaló)
  - Struktúrált kimenetek kinyerése (JSON output)
  - Anthropic Claude API (vagy OpenAI kompatibilis)
  - Fallback: offline NLP megoldások (transformers)

MINTA HASZNÁLATI ESETEK:
  - Vásárlói visszajelzések automatikus elemzése
  - Anomáliák szöveges magyarázata
  - KPI riport automatikus generálása
  - Chatbot üzleti adatokhoz
"""


from __future__ import annotations

# ── Csomagok ellenőrzése ──────────────────────────────────────
try:

    import json
    import os
    import time
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
from pathlib import Path as _Path
OUTPUT_DIR = _Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)




# ════════════════════════════════════════════════════════════
# LLM CLIENT – ABSZTRAKT RÉTEG
# ════════════════════════════════════════════════════════════

class LLMClient:
    """Egyszerű LLM kliens Anthropic Claude-hoz vagy OpenAI-hoz.

    Az API kulcsot environment variable-ből olvassa:
      export ANTHROPIC_API_KEY="sk-..."
      export OPENAI_API_KEY="sk-..."

    Ha nincs API kulcs, szimulált válaszokat ad vissza.
    """

    def __init__(self, provider: str = "anthropic") -> None:
        """
        Args:
            provider: "anthropic" vagy "openai".
        """
        self.provider = provider
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """API kliens inicializálása."""
        if self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                    print("  ✅ Anthropic Claude inicializálva")
                else:
                    print("  ⚠️  ANTHROPIC_API_KEY nem található → szimulált módban fut")
            except ImportError:
                print("  ⚠️  anthropic csomag nem telepítve: pip install anthropic")

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                    print("  ✅ OpenAI inicializálva")
                else:
                    print("  ⚠️  OPENAI_API_KEY nem található → szimulált módban fut")
            except ImportError:
                print("  ⚠️  openai csomag nem telepítve: pip install openai")

    def complete(
        self,
        prompt: str,
        system: str = "Te egy adatelemző asszisztens vagy. Válaszolj tömören, magyarul.",
        max_tokens: int = 1000,
    ) -> str:
        """Szöveges befejezés kérése.

        Args:
            prompt:     A felhasználói üzenet.
            system:     Rendszer-prompt (persona).
            max_tokens: Maximum válaszhossz.

        Returns:
            Az LLM szöveges válasza.
        """
        if self.client is None:
            return self._szimulalt_valasz(prompt)

        try:
            if self.provider == "anthropic":
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"  ⚠️  API hiba: {e}")
            return self._szimulalt_valasz(prompt)

    def _szimulalt_valasz(self, prompt: str) -> str:
        """Offline szimulált válasz (demonstrációs célra)."""
        if "sentiment" in prompt.lower() or "hangulat" in prompt.lower():
            return json.dumps({"sentiment": "pozitív", "score": 0.85, "okok": ["jó minőség", "gyors szállítás"]})
        elif "összefoglaló" in prompt.lower() or "elemezd" in prompt.lower():
            return "A megadott adatok pozitív trendet mutatnak. A bevétel 15%-kal nőtt az előző időszakhoz képest."
        else:
            return "[Szimulált válasz – API kulcs hiányában]"


# ════════════════════════════════════════════════════════════
# 1. VÁSÁRLÓI VISSZAJELZÉSEK ELEMZÉSE
# ════════════════════════════════════════════════════════════

def visszajelzes_elemzes(llm: LLMClient) -> pd.DataFrame:
    """Vásárlói visszajelzések hangulatelemzése LLM-mel.

    PROMPT ENGINEERING TIPP:
      Kérj strukturált JSON kimenetet → könnyű feldolgozás
      Adj konkrét példákat a formátumra (few-shot)
    """
    visszajelzesek = [
        "Kiváló termék, gyors szállítás! Minden rendben volt.",
        "Nagyon csalódott vagyok. A csomag sérülten érkezett és az ügyfélszolgálat sem segített.",
        "Megfelel az árának. Semmi különös, de teljesíti a funkcióját.",
        "Fantasztikus! Már 3-szor rendeltem, mindig elégedett voltam.",
        "Késve érkezett, de a termék maga jó minőségű.",
    ]

    eredmenyek = []

    # Rendszer-prompt: AI szerepének meghatározása
    rendszer_prompt = """Te egy hangulatelemző rendszer vagy.
Minden szöveget elemezz és CSAK JSON formátumban válaszolj, semmi más:
{
  "sentiment": "pozitív" | "negatív" | "semleges",
  "pontszam": 0.0 - 1.0,
  "pozitiv_szempontok": ["..."],
  "negativ_szempontok": ["..."],
  "prioritas": "alacsony" | "közepes" | "magas"
}"""

    print("\n=== VÁSÁRLÓI VISSZAJELZÉS ELEMZÉS ===")

    for i, szoveg in enumerate(visszajelzesek):
        prompt = f'Elemezd ezt a visszajelzést:\n"{szoveg}"'
        valasz = llm.complete(prompt, system=rendszer_prompt)

        try:
            # JSON kinyerése
            parsed = json.loads(valasz)
        except json.JSONDecodeError:
            # Ha nem tiszta JSON, próbálunk kinyerni
            import re
            json_match = re.search(r'\{.*\}', valasz, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except:
                    parsed = {"sentiment": "ismeretlen", "pontszam": 0.5}
            else:
                parsed = {"sentiment": "ismeretlen", "pontszam": 0.5}

        eredmenyek.append({
            "szoveg":    szoveg[:50] + "...",
            "sentiment": parsed.get("sentiment", "?"),
            "pontszam":  parsed.get("pontszam", 0.5),
            "prioritas": parsed.get("prioritas", "?"),
        })

        print(f"  [{i+1}] {parsed.get('sentiment', '?')} "
              f"(score: {parsed.get('pontszam', 0):.2f}): "
              f"{szoveg[:40]}...")

    return pd.DataFrame(eredmenyek)


# ════════════════════════════════════════════════════════════
# 2. AUTOMATIKUS KPI RIPORT
# ════════════════════════════════════════════════════════════

def kpi_riport_generaciо(llm: LLMClient, df: pd.DataFrame) -> str:
    """KPI összefoglaló riport automatikus generálása LLM-mel.

    TIPP: A statisztikákat Python-ban számítsd, ne az LLM-re bízd!
    Az LLM-et csak a szöveges értelmezéshez és prezentációhoz használd.
    """
    # Python számolja a statisztikákat (megbízható!)
    kpik = {
        "atlag_bevetel":     df["bevetel"].mean() if "bevetel" in df else 0,
        "bevetel_valtozas":  "+15.3%",      # placeholder
        "legaktivabb_varos": "Budapest",    # placeholder
        "visszater_arany":   0.64,
    }

    prompt = f"""Az alábbi KPI adatok alapján írj egy rövid (max 150 szó) 
üzleti összefoglalót egy e-commerce cégnek. Emeld ki a pozitív trendeket 
és a figyelmet igénylő területeket:

KPI adatok:
- Átlagos bevétel/tranzakció: {kpik['atlag_bevetel']:,.0f} Ft
- Bevétel változás (MoM): {kpik['bevetel_valtozas']}
- Legjobban teljesítő város: {kpik['legaktivabb_varos']}
- Visszatérő vásárló arány: {kpik['visszater_arany']:.0%}

Stílus: professzionális, de érthető, c-suite szintű közönségnek."""

    riport = llm.complete(prompt)
    print("\n=== AUTOMATIKUS KPI RIPORT ===")
    print(riport)
    return riport


# ════════════════════════════════════════════════════════════
# 3. ANOMÁLIA MAGYARÁZAT
# ════════════════════════════════════════════════════════════

def anomalia_magyarazat(llm: LLMClient) -> None:
    """Anomáliák szöveges magyarázata LLM-mel.

    Workflow:
      1. Python/ML detektálja az anomáliát (számalapú)
      2. LLM értelmezi és magyarázza (kontextus alapján)
    """
    anomalia_adatok = {
        "datum": "2024-03-15",
        "bevetel_drop": -43,   # %
        "normal_nap": "péntek",
        "azonos_het_elso_nap_teljesitmeny": -38,
        "webes_forgalom": "normális",
        "termekkeszlet": "volt hiány: kategória_B",
    }

    prompt = f"""Adatelemzőként magyarázd meg az alábbi anomáliát:
{json.dumps(anomalia_adatok, ensure_ascii=False, indent=2)}

Lehetséges okok (prioritás szerint) és javasolt vizsgálati lépések."""

    print("\n=== ANOMÁLIA MAGYARÁZAT ===")
    magyarazat = llm.complete(prompt)
    print(magyarazat)


# ════════════════════════════════════════════════════════════
# FŐPROGRAM
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    print("=" * 55)
    print("  LECKE 03 – AI + Üzleti Adatelemzés")
    print("=" * 55)
    print("\n  SETUP: Helyezd el az API kulcsot a .env fájlban:")
    print("  ANTHROPIC_API_KEY=sk-...")

    # Kliens inicializálása (API kulcs nélkül szimulált módban fut)
    llm = LLMClient(provider="anthropic")

    # Szintetikus adathalmaz
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "bevetel":  rng.integers(50_000, 500_000, 100),
        "varos":    rng.choice(["Budapest", "Debrecen"], 100),
    })

    # Demók futtatása
    visszajelzes_df = visszajelzes_elemzes(llm)
    print("\nVissszajelzés eredmények:")
    print(visszajelzes_df.to_string(index=False))

    kpi_riport_generaciо(llm, df)
    anomalia_magyarazat(llm)

    # Eredmények mentése
    visszajelzes_df.to_csv(OUTPUT_DIR / "03_visszajelzes_elemzes.csv", index=False, encoding="utf-8")
    print("\n💾 Mentve: outputs/03_visszajelzes_elemzes.csv")

    print("\n✅ Lecke 03 sikeresen lefutott!")
    print("➡️  Következő: 04_ai_onboarding_esettan.py")