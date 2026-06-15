"""
route_optimizer.py  (v2 — Delhi Edition)
-----------------------------------------
Same Greedy Nearest Neighbor algorithm as v1, but distance is now
calculated from real latitude / longitude instead of abstract x / y.

WHY HAVERSINE?
  On a flat grid, simple Pythagoras works fine.
  With lat/lon, the Earth's curvature matters slightly — Haversine gives
  the correct "great-circle" distance between two GPS points.
  It's still just a formula — no external library needed.

DEPOT LOCATION:
  The garbage truck starts from India Gate (28.6129 N, 77.2295 E) —
  a central Delhi landmark that makes a sensible depot.
"""

import numpy as np
import pandas as pd
import math


# ── Depot: India Gate, New Delhi ──────────────────────────────────────────────
DEPOT_LAT = 28.6129
DEPOT_LON = 77.2295
DEPOT_NAME = "India Gate (Depot)"


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculates the straight-line distance (in km) between two GPS points
    using the Haversine formula.

    Parameters:
        lat1, lon1 : Coordinates of point A
        lat2, lon2 : Coordinates of point B

    Returns:
        distance in kilometres (float)
    """
    R = 6371.0   # Earth's radius in km

    # Convert degrees → radians
    phi1, phi2       = math.radians(lat1), math.radians(lat2)
    dphi             = math.radians(lat2 - lat1)
    dlambda          = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def select_active_bins(results_df, threshold=0.6):
    """
    Returns only bins whose overflow probability exceeds the threshold.
    """
    active_df = results_df[results_df["overflow_prob"] > threshold].copy()
    return active_df.reset_index(drop=True)


def greedy_nearest_neighbor(active_df,
                             start_lat=DEPOT_LAT,
                             start_lon=DEPOT_LON):
    """
    Plans the waste collection route using Greedy Nearest Neighbor.

    Starts at the depot (India Gate by default) and always moves to the
    closest unvisited active bin next.

    Parameters:
        active_df  : DataFrame of high-risk bins (with latitude / longitude)
        start_lat  : Depot latitude
        start_lon  : Depot longitude

    Returns:
        route      : List of dicts — one per stop in order
                     Keys: bin_id, location_name, latitude, longitude,
                           distance_km (from previous stop)
        total_km   : Total route distance in kilometres
    """

    if active_df.empty:
        return [], 0.0

    unvisited = active_df[["bin_id", "location_name",
                            "latitude", "longitude"]].to_dict("records")

    current_lat, current_lon = start_lat, start_lon
    route      = []
    total_km   = 0.0

    while unvisited:

        # Distance from current position to every unvisited bin
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

        total_km    += nearest_dist
        current_lat  = nearest_bin["latitude"]
        current_lon  = nearest_bin["longitude"]
        unvisited.pop(nearest_idx)

    return route, round(total_km, 3)


def optimize_route(results_df, threshold=0.6):
    """
    Main entry point: selects active bins and computes the greedy route.

    Returns:
        active_df  : High-risk bins DataFrame
        route      : Ordered stop list
        total_km   : Total route distance in km
    """
    active_df        = select_active_bins(results_df, threshold)
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
