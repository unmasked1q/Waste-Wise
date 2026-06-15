import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# Locations Format:
# (Location Name, Latitude, Longitude, Area Type)
# ─────────────────────────────────────────────────────────────

DELHI_LOCATIONS = [
    ("Connaught Place", 28.6315, 77.2167, "commercial"),
    ("Chandni Chowk", 28.6506, 77.2303, "market"),
    ("Karol Bagh", 28.6527, 77.1886, "market"),
    ("Hauz Khas", 28.5494, 77.2001, "commercial"),
    ("Saket", 28.5244, 77.2066, "commercial"),
    ("Dwarka Sector 21", 28.5511, 77.0565, "residential"),
    ("Lajpat Nagar", 28.5677, 77.2433, "market"),
    ("Nehru Place", 28.5491, 77.2513, "commercial"),
    ("Rajouri Garden", 28.6474, 77.1199, "market"),
    ("Janakpuri", 28.6219, 77.0878, "residential"),
    ("Pitampura", 28.7008, 77.1311, "residential"),
    ("Vasant Kunj", 28.5200, 77.1588, "residential"),
    ("Mayur Vihar", 28.6081, 77.2960, "residential"),
    ("Preet Vihar", 28.6428, 77.3008, "residential"),
    ("Rohini Sector 3", 28.7041, 77.1025, "residential"),
]

# Dataset
def generate_bin_data(num_bins=60, random_seed=42):
    np.random.seed(random_seed)
    rows = []
    for i in range(num_bins):

        # Select random  location
        location = DELHI_LOCATIONS[np.random.randint(0, len(DELHI_LOCATIONS))]

        location_name = location[0]
        base_lat = location[1]
        base_lon = location[2]
        area_type = location[3]

        # Small random variation around actual location
        lat = base_lat + np.random.uniform(-0.005, 0.005)
        lon = base_lon + np.random.uniform(-0.005, 0.005)

        # ---------------------------
        # Fill level generation
        # ---------------------------

        if area_type == "market":
            fill_level = np.random.randint(60, 100)

        elif area_type == "commercial":
            fill_level = np.random.randint(40, 90)

        else:
            fill_level = np.random.randint(20, 80)

        # ---------------------------
        # Overflow history
        # ---------------------------

        if area_type == "market":
            past_overflow_count = np.random.randint(3, 10)

        elif area_type == "commercial":
            past_overflow_count = np.random.randint(1, 7)

        else:
            past_overflow_count = np.random.randint(0, 5)


        overflow = 1 if (
            fill_level > 75 or
            past_overflow_count > 5
        ) else 0

     
        rows.append({
            "bin_id": f"BIN_{i+1}",
            "location_name": location_name,
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "fill_level": fill_level,
            "past_overflow_count": past_overflow_count,
            "area_type": area_type,
            "overflow": overflow
        })

    # DataFrame
    df = pd.DataFrame(rows)
    return df


#---------------------------
# Test
# ---------------------------

if __name__ == "__main__":

    df = generate_bin_data()
    print(df.head())
    print("\nDataset Shape:", df.shape)
    print("\nLocations Covered:")
    print(df["location_name"].unique())
