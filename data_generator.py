"""
data_generator.py  (v2 — Delhi Edition)
----------------------------------------
Generates a realistic dataset of smart waste bins placed across Delhi's
major localities.

WHY NO KAGGLE FILE?
  A direct "Delhi waste bin" Kaggle dataset doesn't exist publicly without
  authentication. Instead, we do what real smart-city projects do when raw
  sensor data isn't yet deployed:
    1. Define real Delhi localities with accurate lat/long coordinates.
    2. Distribute multiple bins per locality (like a real deployment).
    3. Generate realistic fill levels / overflow history using area-specific
       patterns (markets overflow more than residential zones, etc.).

  The ML model, route optimizer, and OpenCV visualizer all work with the
  same latitude / longitude columns — so swapping in a real CSV later only
  requires changing this one file.

COLUMNS PRODUCED
  bin_id              – unique identifier  e.g. "CP_001"
  location_name       – human-readable Delhi area e.g. "Connaught Place"
  latitude            – real GPS latitude  (used for distance & OpenCV)
  longitude           – real GPS longitude (used for distance & OpenCV)
  fill_level          – 0-100 % how full the bin is right now
  past_overflow_count – how many times the bin overflowed this week
  area_type           – "commercial", "market", or "residential"
  overflow            – label: 1 = will overflow soon, 0 = safe
"""

import pandas as pd
import numpy as np


# ── Real Delhi localities with accurate lat/long ───────────────────────────────
# Source: standard GPS coordinates for well-known Delhi landmarks/areas.
# Each entry: (area_name, latitude, longitude, area_type, bins_to_place)
#   area_type drives the fill-level distribution (markets fill faster)
#   bins_to_place controls how many bins are simulated in that locality

DELHI_AREAS = [
    # (name,                   lat,     lon,     type,          bins)
    ("Connaught Place",        28.6315, 77.2167, "commercial",   8),
    ("Chandni Chowk",          28.6506, 77.2303, "market",       8),
    ("Karol Bagh",             28.6527, 77.1886, "market",       7),
    ("Hauz Khas",              28.5494, 77.2001, "commercial",   6),
    ("Saket",                  28.5244, 77.2066, "commercial",   6),
    ("Dwarka Sector 10",       28.5921, 77.0460, "residential",  6),
    ("Lajpat Nagar",           28.5677, 77.2433, "market",       7),
    ("Rohini Sector 3",        28.7041, 77.1025, "residential",  5),
    ("Janakpuri",              28.6219, 77.0878, "residential",  5),
    ("Nehru Place",            28.5491, 77.2513, "commercial",   6),
    ("Pitampura",              28.7008, 77.1311, "residential",  5),
    ("Vasant Kunj",            28.5200, 77.1588, "residential",  5),
    ("Mayur Vihar Phase 1",    28.6081, 77.2960, "residential",  5),
    ("Rajouri Garden",         28.6474, 77.1199, "market",       6),
    ("Preet Vihar",            28.6428, 77.3008, "residential",  5),
]

# Fill-level mean and spread per area type
# Markets and commercial zones fill up much faster than residential areas
FILL_PROFILE = {
    "commercial":  {"mean": 68, "std": 15},
    "market":      {"mean": 78, "std": 12},
    "residential": {"mean": 50, "std": 18},
}

# Overflow history mean per area type (markets overflow more often)
OVERFLOW_HISTORY = {
    "commercial":  {"mean": 2.5, "std": 1.5},
    "market":      {"mean": 4.5, "std": 2.0},
    "residential": {"mean": 1.2, "std": 1.0},
}


def generate_bin_data(num_bins=None, random_seed=42):
    """
    Creates a Delhi-based smart bin dataset.

    Parameters:
        num_bins    : Ignored (kept for API compatibility with old version).
                      Bin count comes from DELHI_AREAS table above (~100 bins).
        random_seed : Fixed seed for reproducibility.

    Returns:
        df : pandas DataFrame with all bin features + overflow label.
    """

    np.random.seed(random_seed)

    rows = []   # We'll build a list of dicts, then convert to DataFrame

    bin_counter = 1   # Global counter for unique bin IDs

    for (area_name, lat, lon, area_type, n_bins) in DELHI_AREAS:

        # Short prefix for bin IDs from this area (first letters of each word)
        prefix = "".join(w[0] for w in area_name.split()[:2]).upper()

        # Area-specific fill level distribution
        fill_mean = FILL_PROFILE[area_type]["mean"]
        fill_std  = FILL_PROFILE[area_type]["std"]

        # Area-specific overflow history distribution
        hist_mean = OVERFLOW_HISTORY[area_type]["mean"]
        hist_std  = OVERFLOW_HISTORY[area_type]["std"]

        for i in range(n_bins):

            # ── Scatter bins within ~500m radius of the area center ──────────
            # 0.005 degrees ≈ 500 m — keeps bins close to their real location
            lat_offset = np.random.uniform(-0.005, 0.005)
            lon_offset = np.random.uniform(-0.005, 0.005)

            bin_lat = round(lat + lat_offset, 6)
            bin_lon = round(lon + lon_offset, 6)

            # ── Fill level ───────────────────────────────────────────────────
            fill_level = int(np.clip(
                np.random.normal(fill_mean, fill_std), 0, 100
            ))

            # ── Past overflow count ──────────────────────────────────────────
            past_overflow = max(0, int(round(
                np.random.normal(hist_mean, hist_std)
            )))

            # ── Overflow label (ground truth for ML training) ────────────────
            # Overflow = 1 when fill is high OR the bin has a bad history
            score = (
                0.6 * (fill_level >= 75)
                + 0.3 * (past_overflow >= 4)
                + 0.1 * np.random.rand()
            )
            overflow = int(score >= 0.5)

            rows.append({
                "bin_id"             : f"{prefix}_{str(i+1).zfill(3)}",
                "location_name"      : area_name,
                "latitude"           : bin_lat,
                "longitude"          : bin_lon,
                "fill_level"         : fill_level,
                "past_overflow_count": past_overflow,
                "area_type"          : area_type,
                "overflow"           : overflow,
            })

            bin_counter += 1

    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


# Quick test
if __name__ == "__main__":
    df = generate_bin_data()
    print(f"Dataset shape : {df.shape}")
    print(f"Areas covered : {df['location_name'].nunique()}")
    print(f"Overflow bins : {df['overflow'].sum()} / {len(df)}")
    print()
    print(df[["bin_id", "location_name", "latitude", "longitude",
              "fill_level", "area_type", "overflow"]].head(12).to_string())
