import numpy as np
import pandas as pd
import math

# Depot: India Gate
DEPOT_LAT = 28.6129
DEPOT_LON = 77.2295
DEPOT_NAME = "India Gate (Depot)"

def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculates dist. b/w 2 points using the Haversine formula.

    lat1, lon1 : Coordinates of point A
    lat2, lon2 : Coordinates of point B
    """
    R = 6371.0   # Earth's radius

    # degrees → radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def select_active_bins(results_df, threshold=0.6):
    active_df = results_df[results_df["overflow_prob"] > threshold].copy()
    return active_df.reset_index(drop=True)


def greedy_nearest_neighbor(active_df, start_lat=DEPOT_LAT, start_lon=DEPOT_LON):

    if active_df.empty:
        return [], 0.0

    unvisited = active_df[["bin_id", "location_name","latitude", "longitude"]].to_dict("records")

    current_lat, current_lon = start_lat, start_lon
    route      = []
    total_km   = 0.0

    while unvisited:

        distances = [
            haversine_km(current_lat, current_lon, b["latitude"], b["longitude"])
            for b in unvisited
        ]

        nearest_idx  = int(np.argmin(distances))
        nearest_bin  = unvisited[nearest_idx]
        nearest_dist = distances[nearest_idx]

        route.append({
            "bin_id"       : nearest_bin["bin_id"],
            "location_name": nearest_bin["location_name"],
            "latitude"     : nearest_bin["latitude"],
            "longitude"    : nearest_bin["longitude"],
            "distance_km"  : round(nearest_dist, 3),
        })

        total_km =total_km+ nearest_dist
        current_lat = nearest_bin["latitude"]
        current_lon = nearest_bin["longitude"]
        unvisited.pop(nearest_idx)

    return route, round(total_km, 3)


def optimize_route(results_df, threshold=0.6):
    active_df = select_active_bins(results_df, threshold)
    route, total_km  = greedy_nearest_neighbor(active_df)
    return active_df, route, total_km


# Quick test
if __name__ == "__main__":
    from data_generator import generate_bin_data
    from model import train_model, predict_overflow

    df = generate_bin_data()
    model, acc, _, _ = train_model(df)
    results_df = predict_overflow(model, df)

    active_df, route, total_km = optimize_route(results_df)
    print(f"Active bins : {len(active_df)}")
    print(f"Total dist  : {total_km:.2f} km")
    print()
    for i, stop in enumerate(route, 1):
        print(f"  Stop {i:2d}: {stop['location_name']:25s} "
              f"({stop['bin_id']})  +{stop['distance_km']:.2f} km")
