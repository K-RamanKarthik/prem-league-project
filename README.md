# ⚽ Premier League Player Scouting & Similarity System (2025-26)

Hey! This is a machine learning scouting dashboard I built for Premier League football analytics. 

It takes player performance stats from the current 2025-26 season, normalizes everything to per-90 metrics, clusters players into distinct playstyle archetypes using K-Means, and lets you find statistical "twins" for any player using Cosine Similarity.

---

## 💡 What it does

- **Live Data Collection**: Automatically pulls data for all 660+ active Premier League players across all 20 teams (supporting both live PL API data and FBref scraping).
- **Per-90 Normalization**: Converts raw stats (goals, xG, assists, xA, key passes, progressive carries, tackles, interceptions) into per-90 metrics so low-minute rotational players can be compared fairly against starters.
- **K-Means Playstyle Clustering**: Groups outfield players into 6 playstyle archetypes (like *Goal-Scoring + Shot-Volume*, *Ball-Progressing + High-Touch*, *Possession-Based*) and goalkeepers into 3 distinct styles.
- **Interactive Cluster Map**: Visualizes player clusters in 2D space using PCA (Principal Component Analysis) with Voronoi region background colors and interactive hover cards.
- **Overlapping Radar Charts**: Compare percentile profiles for multiple players on the same radar plot.
- **Similarity Search**: Calculates cosine similarity on scaled feature vectors to rank top player matches (e.g., finding the closest statistical alternatives to Bukayo Saka, Erling Haaland, or Rodri).

---

## 🛠️ Project Setup & How to Run

First, clone the repo and install the requirements:

```bash
pip install -r requirements.txt
```

### 1. Run the Data & Machine Learning Pipeline
To pull the latest stats, process per-90 features, and train the K-Means & PCA models, run:

```bash
python run_pipeline.py
```

*Note: If you already fetched the data and just want to tweak features or re-train models, you can skip the scraping step:*

```bash
python run_pipeline.py --skip-scrape
```

### 2. Launch the Streamlit Web App
Once the pipeline finishes, start the interactive dashboard:

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`.

---

## 📁 Repository Layout

Here is how the project files are organized:

- `app.py`: Streamlit web dashboard with Plotly radar charts, cluster maps, similarity search, and stat glossaries.
- `run_pipeline.py`: One-command script that runs data collection, feature processing, and ML model training in sequence.
- `scraper.py`: Fetches 2025-26 Premier League player stats across all 20 teams.
- `features.py`: Cleans raw CSVs, converts totals to per-90 stats, and normalizes feature vectors using `StandardScaler`.
- `model.py`: Trains K-Means clustering, PCA dimensionality reduction, and Cosine Similarity models.
- `config.py`: Central config file for feature lists, cluster numbers, paths, and thresholds.
- `data/`: Raw and processed CSV datasets (`data/raw/`, `data/processed/`).
- `models/`: Saved `.pkl` models for scalers and K-Means.

---

## ⚙️ Customization (`config.py`)

You can tweak parameters inside `config.py`:
- `MIN_MINUTES_PLAYED`: Minimum minutes required for a player to be included (set to 45 mins).
- `N_CLUSTERS_OUTFIELD`: Number of playstyle clusters for outfield players (default: 6).
- `N_CLUSTERS_GK`: Number of goalkeeper clusters (default: 3).
- `OUTFIELD_FEATURES` & `GK_FEATURES`: The specific metrics used for training the models.

---

## 🧰 Tech Stack

- **Framework & Dashboard**: Streamlit, Plotly
- **Machine Learning**: scikit-learn (KMeans, PCA, StandardScaler, Cosine Similarity)
- **Data Wrangling**: pandas, numpy
- **Web Scraping & APIs**: requests, BeautifulSoup, curl_cffi

Feel free to check it out, tweak the cluster numbers in `config.py`, or deploy it on Streamlit Cloud!
