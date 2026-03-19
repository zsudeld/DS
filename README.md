# 🎓 Data Science & Machine Learning Kurzus

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Leckék-18%20fájl-green" />
  <img src="https://img.shields.io/badge/Kód-~5900%20sor-lightgrey" />
  <img src="https://img.shields.io/badge/Nyelv-Magyar-red" />
</p>

<p align="center">
  <b>Teljes körű, magyar nyelvű, gyakorlati Data Science képzés – futtatható Python boilerplate kódokkal.</b><br/>
  A Python alapjaitól egészen a modellek éles rendszerbe helyezéséig.
</p>

---

## 📋 Tartalomjegyzék

- [Kurzusról](#-kurzusról)
- [Indítás](#-indítás)
- [Kurzustérkép](#-kurzustérkép)
- [Tanulási sorrend](#-tanulási-sorrend)
- [Fájlok leírása](#-fájlok-leírása)
- [Design elvek](#-design-elvek)
- [Projekt struktúra](#-projekt-struktúra)
- [Követelmények](#-követelmények)

---

## 📖 Kurzusról

Ez a repó egy **teljes körű, gyakorlatorientált Data Science kurzus**, amely 18 önállóan futtatható Python leckéből áll. Minden fájl egy valós üzleti problémát old meg **szintetikus adatokon** – nincs szükség külső adatletöltésre, minden adat a kód futtatásakor generálódik.

### Miben más ez, mint egy online kurzus?

| Online kurzus | Ez a repó |
|---|---|
| Videók és slideok | Futtatható, kommentált Python kód |
| Elmagyarázza *mit* csinál | Elmagyarázza *miért* úgy csinál |
| Általános példák | Valós üzleti kontextus |
| Eszközönként külön | Teljes pipeline egy helyen |

### Kinek szól?

- 🐍 Programozási alapokkal rendelkező, DS-ben kezdő fejlesztőknek
- 📊 Junior adatelemzőknek, akik strukturált áttekintést keresnek
- 🤖 ML mérnököknek, akik gyors referencia-kódot keresnek

---

## ⚡ Indítás

```bash
# 1. Klónozd a repót
git clone https://github.com/zsudeld/DS_kurzus.git
cd ds_kurzus

# 2. Hozz létre virtuális környezetet (ajánlott)
python -m venv .venv
.venv\Scripts\activate           # Windows

# 3. Telepítsd a függőségeket
pip install -r requirements.txt

# 4. Ellenőrizd a telepítést
python telepites_ellenorzes.py

# 5. Futtasd az első leckét
python leckek/00_vizualizacio_alapok.py
```

> **Megjegyzés:** A generált grafikonok és kimeneti fájlok az `outputs/` mappában jelennek meg.

---

## 🗺️ Kurzustérkép

### 📊 BLOKK 1 – Vizualizáció alapok
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`00_vizualizacio_alapok.py`](leckek/00_vizualizacio_alapok.py) | Matplotlib & Seaborn | Figure/Axes, subplot, vonal/oszlop/hisztogram/scatter, boxplot, violin, heatmap, pairplot, KDE |

---

### 🐍 BLOKK 2 – Python és Adatkezelés alapok
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`01_python_intro.py`](leckek/01_python_intro.py) | Python alapok | Lista, dict, set, comprehension, lambda, map/filter/zip, NumPy, Pandas DataFrame, merge, groupby |

---

### 🧹 BLOKK 3 – Adatkezelés, Feature Engineering, SQL
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`02_adattisztitas.py`](leckek/02_adattisztitas.py) | Adattisztítás | NaN kezelés, IQR/Z-score outlier, duplikátumok, dtype javítás |
| [`08_ml_adatelokeszites.py`](leckek/08_ml_adatelokeszites.py) | ML előkészítés | StandardScaler, OneHotEncoder, ColumnTransformer, sklearn Pipeline |
| [`10_ml_feature_engineering1.py`](leckek/10_ml_feature_engineering1.py) | Feature Eng. I. | Domain jellemzők, binning, log/sqrt transzformáció, interakció-tagok |
| [`11_ml_feature_engineering2.py`](leckek/11_ml_feature_engineering2.py) | Feature Eng. II. | Idősor-jellemzők, rolling window, target encoding, feature selection |
| [`24_sql_adatelers.py`](leckek/24_sql_adatelers.py) | SQL & Adatelérés | SQLite, SELECT/WHERE/JOIN, GROUP BY, ablakfüggvények, Pandas↔SQL |

---

### 📐 BLOKK 4 – Statisztika
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`20_21_22_23_statisztika.py`](leckek/20_21_22_23_statisztika.py) | Statisztika | Pearson/Spearman korreláció, OLS, normalitásteszt, t-teszt, ANOVA, Chi² |

---

### 📈 BLOKK 5 – Haladó vizualizáció
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`18_plotly_express.py`](leckek/18_plotly_express.py) | Plotly Express | Interaktív scatter, bar, box, heatmap, animáció, HTML dashboard export |

---

### 🤖 BLOKK 6 – Machine Learning Pipeline
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`14_15_16_ml_regresszio_workflow.py`](leckek/14_15_16_ml_regresszio_workflow.py) | Regresszió & Osztályozás | Ridge/Lasso/ElasticNet, logisztikus reg., ROC-AUC, confusion matrix |
| [`17_ml_train_test_split.py`](leckek/17_ml_train_test_split.py) | Kiértékelés | StratifiedKFold, TimeSeriesSplit, learning curve, data leakage |
| [`07_12_13_ensemble_klaszter.py`](leckek/07_12_13_ensemble_klaszter.py) | Ensemble & Klaszter | Random Forest, XGBoost, LightGBM, CatBoost, K-Means, DBSCAN |
| [`09_19_automl_tracking.py`](leckek/09_19_automl_tracking.py) | AutoML & Tracking | Optuna (Bayesiánus HPO), FLAML AutoML, MLflow experiment tracking |

---

### 🕐 BLOKK 7 – Idősor & NLP
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`25_idosor_elorejelzes.py`](leckek/25_idosor_elorejelzes.py) | Idősor-előrejelzés | STL dekompozíció, ADF-teszt, ARIMA, Prophet, MAE/RMSE/MAPE |
| [`26_nlp_alapok.py`](leckek/26_nlp_alapok.py) | NLP alapok | Tokenizáció, TF-IDF, szövegklasszifikáció, szóbeágyazás |

---

### 🧠 BLOKK 8 – AI Integráció & Deployment
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`03_ai_databiz.py`](leckek/03_ai_databiz.py) | LLM integráció | Anthropic/OpenAI API, JSON output, KPI riport, anomália magyarázat |
| [`27_deployment_mlops.py`](leckek/27_deployment_mlops.py) | Deployment & MLOps | FastAPI REST API, joblib, Dockerfile, docker-compose, PSI/KS drift |

---

### 💼 BLOKK 9 – Esettanulmányok
| Fájl | Témakör | Kulcsfogalmak |
|------|---------|---------------|
| [`04_05_06_esettanulmanyok.py`](leckek/04_05_06_esettanulmanyok.py) | 3 komplex eset | HR előrejelzés (SHAP) · Kripto bot (backtesting, Sharpe) · Hálózatelemzés (NetworkX, PageRank) |

---

## 🧭 Tanulási sorrend

```
[00] Vizualizáció alapok (Matplotlib/Seaborn)
  │
  ▼
[01] Python · NumPy · Pandas
  │
  ├──▶ [02] Adattisztítás ──▶ [08] Pipeline ──▶ [10] [11] Feature Engineering
  │
  ├──▶ [24] SQL & Adatelérés
  │
  ├──▶ [20–23] Statisztika
  │
  ▼
[18] Interaktív vizualizáció (Plotly)
  │
  ▼
[17] Kiértékelési stratégiák (CV · leakage)
  │
  ├──▶ [14–16] Regresszió & Osztályozás
  ├──▶ [07/12/13] Ensemble modellek & Klaszterezés
  └──▶ [09/19] AutoML & MLflow
  │
  ├──▶ [25] Idősor-előrejelzés (ARIMA · Prophet)
  └──▶ [26] NLP (TF-IDF · szövegklasszifikáció)
  │
  ├──▶ [03] LLM / AI API integráció
  └──▶ [27] Deployment: FastAPI · Docker · Monitoring
  │
  ▼
[04–06] Esettanulmányok  🎉
```

---

## 📂 Fájlok leírása

<details>
<summary><b>00 – Vizualizáció alapok</b> (kattints a részletekért)</summary>

Matplotlib és Seaborn alapok nulláról. Megtanítja a Figure/Axes objektummodellt, az alap diagramtípusokat (vonal, oszlop, hisztogram, scatter, kördiagram), majd a statisztikai vizualizációkat (boxplot, violin plot, heatmap, regressziós scatter, KDE). A pairplot szekcióban bemutatja, hogyan lehet egyszerre több változópár kapcsolatát vizsgálni. Minden diagram mentve lesz az `outputs/` mappába.
</details>

<details>
<summary><b>01 – Python, NumPy, Pandas</b></summary>

Python adatstruktúrák (lista, dict, set, tuple) és funkcionális eszközök (lambda, map, filter, zip, comprehension). Második felében NumPy tömbök, indexelés, broadcasting, majd Pandas DataFrame létrehozás, szűrés, merge és groupby műveletek.
</details>

<details>
<summary><b>02 – Adattisztítás</b></summary>

Valósághű, szándékosan „piszkos" szintetikus adathalmazon mutatja be a teljes adattisztítási folyamatot: hiányzó értékek azonosítása és pótlása (mean/median/mode/konstans), IQR- és Z-score-alapú outlier detektálás, duplikátumok kezelése, dtype-hibák javítása, kategorikus konzisztencia. Vizualizációkkal támogatva.
</details>

<details>
<summary><b>08 – ML előkészítés (Pipeline)</b></summary>

Az sklearn Pipeline és ColumnTransformer mélységes bemutatása. Numerikus jellemzőkre StandardScaler/RobustScaler, kategorikus jellemzőkre OneHotEncoder/OrdinalEncoder. Megmutatja, miért kell a teljes Pipeline-t menteni (nem csak a modellt), és hogyan előzhetők meg a data leakage hibák.
</details>

<details>
<summary><b>10 – Feature Engineering I.</b></summary>

Klasszikus feature engineering technikák: domain-specifikus jellemzők kiszámítása, binning (egyenlő szélességű és kvantilis-alapú), logaritmikus és négyzetgyök transzformáció ferdeség csökkentésére, interakció-tagok és polinomiális jellemzők létrehozása.
</details>

<details>
<summary><b>11 – Feature Engineering II.</b></summary>

Idősor-alapú jellemzők (lag, rolling mean/std, dátumkomponensek), target encoding (leave-one-out változattal, data leakage figyelemmel), majd feature selection: univariáns (SelectKBest), modell-alapú (feature_importances_) és rekurzív eliminációs (RFE) módszerek összehasonlítása.
</details>

<details>
<summary><b>14–16 – Regresszió & Osztályozás & ML Workflow</b></summary>

Három témakör egy fájlban: (1) Lineáris regresszió Ridge/Lasso/ElasticNet regularizációval, keresztvalidációval. (2) Logisztikus regresszió bináris és multi-class esetekre, ROC-AUC görbe, osztályozási küszöb optimalizálása. (3) Összefoglaló táblázat: mikor melyik modellt érdemes választani.
</details>

<details>
<summary><b>17 – Train/Test Split & Kiértékelés</b></summary>

Az egyik legfontosabb fájl: a kiértékelési stratégiák helyes alkalmazása. Train/test/validation hármas split, StratifiedKFold (osztályozáshoz), TimeSeriesSplit (idősorokhoz), learning curve elemzés (alul- vs. túlillesztés diagnózis), data leakage részletes bemutatása hibás és helyes példával.
</details>

<details>
<summary><b>07/12/13 – Ensemble modellek & Klaszterezés</b></summary>

Random Forest: ensemble elmélet, out-of-bag hiba, feature importance, RandomizedSearchCV. XGBoost vs. LightGBM vs. CatBoost összehasonlítás egyazon adathalmazon. Felügyelet nélküli tanulás: K-Means (könyök-módszer, sziluett-pontszám), DBSCAN (zajrobusztus), hierarchikus klaszterezés dendrogrammal.
</details>

<details>
<summary><b>09/19 – AutoML & MLflow Tracking</b></summary>

Optuna: Bayesiánus hyperparaméter-optimalizálás, study objektum, pruning, vizualizáció. FLAML AutoML: minimális kóddal versenyképes modell. MLflow: experiment tracking, run logolás (paraméterek, metrikák, artifactek), modell-verziókezelés, UI elindítása.
</details>

<details>
<summary><b>18 – Plotly Express (interaktív vizualizáció)</b></summary>

Interaktív vizualizációk Plotly Express-szel: scatter (hover, szín, méret), bar (animáció), box, idősor, heatmap, facet grid. A fájl végén egy önálló HTML dashboardot generál, amelyet böngészőben meg lehet nyitni és interaktívan használni.
</details>

<details>
<summary><b>20–23 – Statisztika</b></summary>

Statisztikai alapok data science kontextusban: Pearson és Spearman korreláció (mikor melyik), korreláció-hőtérkép. OLS lineáris regresszió statsmodels-szel (R², p-értékek, feltételek ellenőrzése). Normalitástesztek (Shapiro-Wilk, Q-Q plot). Hipotézisvizsgálatok: t-teszt, egyutas ANOVA (post-hoc Tukey), Chi-négyzet függetlenségvizsgálat.
</details>

<details>
<summary><b>24 – SQL & Adatelérés</b></summary>

SQL alapok Python adattudós szemszögéből. Valódi SQLite adatbázist hoz létre (alkalmazottak, termékek, értékesítések táblák). Bemutatja a SELECT/WHERE alapokat, GROUP BY + aggregáció, INNER vs. LEFT JOIN, HAVING szűrés, majd ablakfüggvények (RANK, gördülő összeg). Pandas `read_sql` és `to_sql` integráció.
</details>

<details>
<summary><b>25 – Idősor-előrejelzés</b></summary>

Szintetikus napi látogatószám-idősor (trend + heti + éves szezonalitás + zaj). STL dekompozíció vizualizációval. ADF stacionaritásteszt. ARIMA(1,1,1) heti aggregált adatokon, 8 hetes előrejelzéssel és konfidencia-intervallummal. Prophet modell napi adatokon, magyar ünnepnapokkal, changepoint detektálással. MAE/RMSE/MAPE összehasonlítás.
</details>

<details>
<summary><b>26 – NLP alapok</b></summary>

Magyar szöveges adathalmazon (szintetikus ügyfélvélemények) végigvezeti a klasszikus NLP pipeline-t: szöveg normalizálás, tokenizáció, stopszavak, TF-IDF vektorizáció bigramokkal. Három modell összehasonlítása cross-validációval (Logisztikus Regresszió, Naív Bayes, LinearSVC). Confusion matrix és jellemző szó vizualizáció. Szóbeágyazás (Word2Vec/sentence-transformers) elméleti összefoglaló.
</details>

<details>
<summary><b>03 – AI / LLM integráció</b></summary>

Anthropic Claude és OpenAI API integráció üzleti adatelemzéshez. JSON-t visszaadó promptok strukturált kimenettel. Sentiment elemzés LLM-mel, összehasonlítva a klasszikus ML megközelítéssel. KPI riport automatikus generálása. Anomália detektálás és magyarázat LLM-mel.
</details>

<details>
<summary><b>27 – Deployment & MLOps</b></summary>

A teljes ML deployment ciklus: modell betanítás és mentés joblib-bal (Pipeline-t ment, nem csak a modellt), model card JSON metaadatokkal. Teljes, futtatható FastAPI alkalmazás (Pydantic input validáció, /health, /predict, /predict/batch). Dockerfile és docker-compose leírás. Data drift detektálás PSI (Population Stability Index) és Kolmogorov-Smirnov teszttel. Deployment checklist.
</details>

<details>
<summary><b>04–06 – Esettanulmányok</b></summary>

Három komplex, end-to-end üzleti eset. (1) **HR:** onboarding sikeresség előrejelzése Random Forest-tel, SHAP értékekkel. (2) **Kripto:** árfolyam-előrejelzés LightGBM-mel, backtesting motorral, Sharpe-ráta és drawdown számítással. (3) **Hálózat:** vállalati kommunikációs gráf NetworkX-szel, PageRank rangsorolás, közösségdetektálás.
</details>

---

## ✅ Design elvek

**🔁 Önálló futtathatóság**
Minden fájl `python leckek/fajlnev.py` paranccsal fut. Nincs szükség külső adatokra.

**🛡️ Graceful degradation**
Ha egy opcionális csomag (pl. `xgboost`, `mlflow`, `prophet`) nincs telepítve, a kód figyelmeztetést ír ki és fut tovább.

**🇭🇺 Magyar dokumentáció**
Minden függvénynek van docstring-je, amely elmagyarázza *mit* és *miért*. Az inline kommentek a nem triviális lépéseket magyarázzák.

**⚠️ Data leakage tudatosság**
Kiemelten jelölve minden hely, ahol data leakage fordulhat elő – helyes és hibás példával egymás mellett.

**🔢 Reprodukálhatóság**
Minden fájlban `np.random.seed(42)` biztosítja az eredmények reprodukálhatóságát.

**📝 Type hints**
Modern Python 3.10+ stílus: minden függvénynek van típus annotációja.

---

## 📁 Projekt struktúra

```
ds-kurzus/
│
├── README.md                           ← Ez a fájl
├── requirements.txt                    ← Összes függőség
├── .gitignore
│
├── leckek/
│   ├── 00_vizualizacio_alapok.py       ← Matplotlib & Seaborn
│   ├── 01_python_intro.py              ← Python, NumPy, Pandas
│   ├── 02_adattisztitas.py             ← Adattisztítás
│   ├── 03_ai_databiz.py                ← LLM API integráció
│   ├── 04_05_06_esettanulmanyok.py     ← HR · Kripto · Hálózat
│   ├── 07_12_13_ensemble_klaszter.py   ← RF · XGB · Klaszter
│   ├── 08_ml_adatelokeszites.py        ← Pipeline, Encoding
│   ├── 09_19_automl_tracking.py        ← Optuna, FLAML, MLflow
│   ├── 10_ml_feature_engineering1.py   ← Feature Eng. I.
│   ├── 11_ml_feature_engineering2.py   ← Feature Eng. II.
│   ├── 14_15_16_ml_regresszio_workflow.py
│   ├── 17_ml_train_test_split.py       ← CV, leakage
│   ├── 18_plotly_express.py            ← Interaktív vizualizáció
│   ├── 20_21_22_23_statisztika.py      ← Stat. tesztek
│   ├── 24_sql_adatelers.py             ← SQL & adatelérés
│   ├── 25_idosor_elorejelzes.py        ← ARIMA & Prophet
│   ├── 26_nlp_alapok.py                ← TF-IDF & NLP
│   └── 27_deployment_mlops.py          ← FastAPI, Docker, drift
│
└── outputs/                            ← Auto-generált kimenetek (gitignore-ban)
    └── .gitkeep
```

---

## 📦 Követelmények

A teljes `requirements.txt` tartalmaz minden szükséges csomagot. A főbb függőségek:

```
numpy>=1.26
pandas>=2.1
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
plotly>=5.18
xgboost>=2.0
lightgbm>=4.2
catboost>=1.2
statsmodels>=0.14
optuna>=3.5
flaml>=2.1
mlflow>=2.10
prophet>=1.1
fastapi>=0.110
uvicorn>=0.27
joblib>=1.3
anthropic>=0.20
openai>=1.12
networkx>=3.2
scipy>=1.12
pingouin>=0.5
```

---


*Utolsó frissítés: 2026 · Python 3.10+ · scikit-learn 1.4+*
