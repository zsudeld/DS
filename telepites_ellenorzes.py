"""
╔══════════════════════════════════════════════════════════════╗
║  Telepítés-ellenőrző                                         ║
║  Futtatás: python telepites_ellenorzes.py                    ║
╚══════════════════════════════════════════════════════════════╝

Ellenőrzi, hogy minden szükséges csomag telepítve van-e.
Ha valami hiányzik, megmutatja a telepítési parancsot.
"""

import sys
import subprocess

# (csomagnév importban, pip-csomagnév, kötelező-e?)
CSOMAGOK = [
    # Kötelező – nélkülük semmi sem fut
    ("numpy",         "numpy",            True),
    ("pandas",        "pandas",           True),
    ("matplotlib",    "matplotlib",       True),
    ("seaborn",       "seaborn",          True),
    ("sklearn",       "scikit-learn",     True),
    ("scipy",         "scipy",            True),
    ("joblib",        "joblib",           True),

    # Ajánlott – a legtöbb leckéhez kell
    ("statsmodels",   "statsmodels",      False),
    ("plotly",        "plotly",           False),
    ("xgboost",       "xgboost",          False),
    ("lightgbm",      "lightgbm",         False),
    ("catboost",      "catboost",         False),
    ("optuna",        "optuna",           False),
    ("flaml",         "flaml",            False),
    ("mlflow",        "mlflow",           False),
    ("networkx",      "networkx",         False),
    ("pingouin",      "pingouin",         False),

    # Opcionális – csak egyes leckékhez
    ("prophet",       "prophet",          False),
    ("fastapi",       "fastapi",          False),
    ("uvicorn",       "uvicorn",          False),
    ("pydantic",      "pydantic",         False),
    ("anthropic",     "anthropic",        False),
    ("openai",        "openai",           False),
]

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ellenorzes():
    print(f"\n{BOLD}{'='*60}")
    print("  Telepítés-ellenőrző – DS Kurzus")
    print(f"{'='*60}{RESET}")
    print(f"  Python: {sys.version.split()[0]}  ({sys.executable})\n")

    hianyzo_kotelezo   = []
    hianyzo_ajanlott   = []
    hianyzo_opcionalis = []

    for import_nev, pip_nev, kotelezo in CSOMAGOK:
        try:
            __import__(import_nev)
            # Verzió lekérdezése
            mod = sys.modules[import_nev]
            verzio = getattr(mod, "__version__", "?")
            print(f"  {GREEN}✅{RESET}  {import_nev:<20} {CYAN}v{verzio}{RESET}")
        except ImportError:
            cimke = "KÖTELEZŐ" if kotelezo else ("ajánlott" if pip_nev not in ("prophet","fastapi","uvicorn","pydantic","anthropic","openai") else "opcionális")
            szin  = RED if kotelezo else YELLOW
            print(f"  {RED}❌{RESET}  {import_nev:<20} {szin}[{cimke}]{RESET}  → pip install {pip_nev}")
            if kotelezo:
                hianyzo_kotelezo.append(pip_nev)
            elif cimke == "ajánlott":
                hianyzo_ajanlott.append(pip_nev)
            else:
                hianyzo_opcionalis.append(pip_nev)

    print(f"\n{BOLD}{'─'*60}{RESET}")

    if not hianyzo_kotelezo and not hianyzo_ajanlott and not hianyzo_opcionalis:
        print(f"\n  {GREEN}{BOLD}🎉  Minden csomag telepítve! Készen állsz a kurzusra.{RESET}")
    else:
        if hianyzo_kotelezo:
            print(f"\n  {RED}{BOLD}❌  Kötelező hiányzó csomagok:{RESET}")
            print(f"     pip install {' '.join(hianyzo_kotelezo)}")

        if hianyzo_ajanlott:
            print(f"\n  {YELLOW}⚠️   Ajánlott hiányzó csomagok:{RESET}")
            print(f"     pip install {' '.join(hianyzo_ajanlott)}")

        if hianyzo_opcionalis:
            print(f"\n  ℹ️   Opcionális (csak egyes leckékhez):")
            print(f"     pip install {' '.join(hianyzo_opcionalis)}")

        print(f"\n  {CYAN}💡  Az összes egyszerre:{RESET}")
        print(f"     pip install -r requirements.txt\n")

        # Automatikus telepítés ajánlata
        if hianyzo_kotelezo:
            valasz = input(f"\n  Telepítsem most a kötelező csomagokat? [i/n] ").strip().lower()
            if valasz == "i":
                print(f"\n  Telepítés: pip install {' '.join(hianyzo_kotelezo)}\n")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + hianyzo_kotelezo,
                    check=False
                )
                print("\n  Futtasd újra a szkriptet az ellenőrzéshez.")

    print()


if __name__ == "__main__":
    ellenorzes()
