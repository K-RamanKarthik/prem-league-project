# ⚽ Premier League Player Scouting & Similarity System (2025-26)

Hey! This is an AI/ML-powered football scouting & analytics application built for the Premier League 2025-26 season.

It processes performance stats for all 660+ active Premier League players across all 20 teams, normalizes metrics per-90 minutes, clusters players into playstyle archetypes using **K-Means**, and finds statistical equivalents for any player using **Cosine Similarity**.

---

## 💡 Key Features & Upgrades

- **Complete Premier League Squad Database**: Includes all 667 active Premier League players (594 outfield players + 73 goalkeepers across 20 teams) with `MIN_MINUTES_PLAYED = 0`.
- **Per-90 Normalization**: Converts raw totals into per-90 metrics (goals, xG, assists, xA, key passes, progressive carries/passes, tackles, interceptions) for fair comparisons.
- **K-Means Playstyle Clustering**: Clusters outfield players into 6 playstyle archetypes (e.g., *Goal-Scoring + Goal-Creating*, *Possession-Based Playmaking*, *Progressive Passing*, *Aerial + Dribbling*) and goalkeepers into distinct styles.
- **Interactive 2D PCA Cluster Map**: Visualizes playstyle clusters in 2D space with color-matched translucent **Convex Hull Bubbles**, **Continuous KNN Decision Grid** options, **scroll-to-zoom**, and **click-and-drag pan**.
- **Overlapping Multi-Player Radars**: Compare percentile profiles for up to 4 players simultaneously on a single radar chart.
- **Cosine Similarity Matcher**: Ranks top statistical equivalents (e.g., finding alternatives to Bukayo Saka, Erling Haaland, or Rodri).
- **Tactical Roles & Stat Guide**: Includes a comprehensive guide explaining position abbreviations (**GK**, **DF/CB**, **FB/WB**, **MF/DM/CM**, **AM/#10**, **FW/W/ST**), tactical roles, and per-90 metric definitions.
- **European Character & Accent Support**: Full UTF-8 string decoding (*Martin Ødegaard*, *Jurriën Timber*, *Gabriel dos Santos Magalhães*, *Victor Lindelöf*, *Pascal Groß*) styled with Google Fonts (**Inter** & **Outfit**).

---

> [!NOTE]  
> **Player Dropdown Ordering & Aaron Anselmino Note**:  
> Player selection dropdowns prioritize active players with >0 minutes played in alphabetical order. This ensures that **Aaron Hickey** (Brentford, 742 mins) is the #1 default player at index 0 when the dashboard opens, displaying full radar charts and stats right away.  
>   
> **Aaron Anselmino** (and other players with 0 minutes played in the 2025-26 season) has been moved to the **bottom of the player list** so 0-stat profiles don't block the default view, while remaining 100% searchable and accessible at the end of the list.

---

## 🛠️ Project Setup & How to Run

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 1. Run Data Scraping & ML Pipeline
To fetch data, extract per-90 features, and train the K-Means & PCA models, run:

```bash
python run_pipeline.py
```

*To re-train ML models or update features without re-scraping live data, use:*

```bash
python run_pipeline.py --skip-scrape
```

### 2. Launch the Streamlit Dashboard
Start the web app:

```bash
streamlit run app.py
```

Open `http://localhost:8501` (or `http://localhost:8510`) in your browser.

---

## 📁 Repository Structure

- `app.py`: Streamlit web dashboard featuring overlapping radar charts, remade PCA cluster maps, similarity matcher, and tactical guides.
- `run_pipeline.py`: Pipeline executor running data collection, feature engineering, and model training in sequence.
- `scraper.py`: Fetches 2025-26 Premier League data via live FPL API with FBref fallback.
- `features.py`: Cleans raw CSVs, calculates per-90 metrics, and normalizes feature vectors using `StandardScaler`.
- `model.py`: Fits K-Means clustering, PCA 2D projections, and Cosine Similarity models.
- `config.py`: Central configuration for feature lists, cluster parameters, paths, and thresholds.
- `data/`: Raw and processed dataset files (`data/raw/`, `data/processed/`).
- `models/`: Saved `.pkl` model files for scalers, K-Means, and PCA models.

---

## 🧰 Tech Stack

- **Dashboard & UI**: Streamlit, Plotly, HTML/CSS (Inter & Outfit Google Fonts)
- **Machine Learning**: scikit-learn (KMeans, PCA, StandardScaler, Cosine Similarity), scipy
- **Data Processing**: pandas, numpy
- **Data Scraping & APIs**: requests, BeautifulSoup, curl_cffi

---

## 👨‍💻 Author & Connect

Created by **Kotamraju Raman Karthik**

- 🐙 **GitHub Repository**: [K-RamanKarthik/prem-league-project](https://github.com/K-RamanKarthik/prem-league-project)
- 💼 **LinkedIn Profile**: [Kotamraju Raman Karthik](https://www.linkedin.com/in/kotamraju-raman-karthik/)

Feel free to star the repository or connect on LinkedIn!
