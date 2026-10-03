"""
Configuration for the Premier League Player Scouting System.
"""
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Scraping ---
FBREF_BASE_URL = "https://fbref.com"
SEASON = "2025-2026"
COMPETITION = "Premier-League"
COMP_ID = "9"  # FBref competition ID for Premier League

# Request headers — must look like a real browser to pass Cloudflare
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}
REQUEST_DELAY = 6  # seconds between requests (FBref max = 10 req/min)

# --- Feature Engineering ---
# Minimum minutes played to include a player
MIN_MINUTES_PLAYED = 45  # Filter out players with less than 45 mins

# Stats to use for clustering (per 90 minutes where applicable)
OUTFIELD_FEATURES = [
    "goals_per90",
    "assists_per90",
    "xg_per90",
    "xa_per90",
    "shots_per90",
    "shots_on_target_per90",
    "passes_completed_per90",
    "pass_completion_pct",
    "key_passes_per90",
    "progressive_passes_per90",
    "progressive_carries_per90",
    "successful_dribbles_per90",
    "tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    "aerials_won_per90",
    "sca_per90",
    "gca_per90",
    "touches_per90",
    "carries_per90",
]

GK_FEATURES = [
    "saves_per90",
    "save_pct",
    "clean_sheet_pct",
    "goals_against_per90",
    "psxg_minus_ga",  # post-shot xG minus goals allowed (shot-stopping quality)
    "passes_completed_pct_gk",
    "passes_launched_pct",
    "crosses_stopped_pct",
    "avg_distance_defensive_actions",
]

# --- Clustering ---
N_CLUSTERS_OUTFIELD = 6  # Number of clusters for outfield players
N_CLUSTERS_GK = 3  # Number of clusters for goalkeepers
RANDOM_STATE = 42

# Cluster labels (will be assigned after analyzing cluster centroids)
OUTFIELD_CLUSTER_LABELS = {
    0: "Goal Poacher",
    1: "Creative Playmaker",
    2: "Box-to-Box Engine",
    3: "Defensive Anchor",
    4: "Wide Threat",
    5: "Ball-Playing Defender",
}

GK_CLUSTER_LABELS = {
    0: "Shot-Stopper",
    1: "Sweeper-Keeper",
    2: "Traditional Keeper",
}

# --- Similarity Search ---
N_SIMILAR_PLAYERS = 10  # Number of similar players to return

# --- Dashboard ---
STREAMLIT_PAGE_TITLE = "⚽ PL Player Scout"
STREAMLIT_PAGE_ICON = "⚽"
STREAMLIT_LAYOUT = "wide"
