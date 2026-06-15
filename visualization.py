"""
visualization.py  (v2 — Delhi Edition)
----------------------------------------
Draws a Delhi-style city map using OpenCV.

WHAT CHANGED FROM v1:
  - Coordinates are now real lat/lon (Delhi bounding box ~28.40–28.75 N,
    76.85–77.35 E) instead of a 0–100 abstract grid.
  - Bins are LABELLED with their location_name (e.g. "Connaught Place")
    instead of raw coordinate numbers.
  - The depot is shown as India Gate with a proper label.
  - A compass rose (N arrow) is drawn for map authenticity.
  - Route distances now shown in kilometres.

The canvas projection is a simple linear mapping:
    pixel_x = scale(longitude)
    pixel_y = scale(latitude)   ← flipped so North is up
This is good enough for a city-scale map without any GIS library.
"""

import cv2
import numpy as np
from route_optimizer import DEPOT_LAT, DEPOT_LON, DEPOT_NAME


# ── Canvas dimensions ──────────────────────────────────────────────────────────
CANVAS_W = 1000
CANVAS_H = 850
PAD      = 70     # Padding so labels near the edge aren't clipped

# ── Delhi bounding box (defines what region the map covers) ───────────────────
# These values just need to contain all Delhi areas we use — not exact borders.
LAT_MIN, LAT_MAX = 28.40, 28.80   # South to North
LON_MIN, LON_MAX = 76.85, 77.38   # West to East


def latlon_to_pixel(lat, lon):
    """
    Converts a GPS (lat, lon) point to pixel (x, y) on the canvas.

    Longitude → x  (increases left → right, West → East)
    Latitude  → y  (FLIPPED: increases bottom → top, South → North)
    """
    px = int(PAD + (lon - LON_MIN) / (LON_MAX - LON_MIN) * (CANVAS_W - 2 * PAD))
    py = int(CANVAS_H - PAD - (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (CANVAS_H - 2 * PAD))
    return px, py


def _short_label(location_name):
    """
    Shortens a long location name to fit on the map.
    e.g. 'Dwarka Sector 10' → 'Dwarka S.10'
         'Mayur Vihar Phase 1' → 'Mayur Vihar'
    Returns at most 14 characters.
    """
    parts = location_name.split()
    if len(parts) == 1:
        return location_name[:14]
    return " ".join(parts[:2])[:14]


def draw_route_map(results_df, active_df, route):
    """
    Renders the Delhi bin map with route overlay.

    Parameters:
        results_df : Full predictions DataFrame (all bins)
        active_df  : High-risk bins only
        route      : Ordered stop list from route_optimizer

    Returns:
        img_rgb : H×W×3 numpy array in RGB colour order (for st.image)
    """

    # ── 1. Dark canvas ─────────────────────────────────────────────────────────
    img = np.full((CANVAS_H, CANVAS_W, 3), (15, 20, 35), dtype=np.uint8)

    # ── 2. Subtle lat/lon grid ─────────────────────────────────────────────────
    grid_col = (30, 38, 58)

    # Vertical lines every 0.05° longitude
    lon = LON_MIN
    while lon <= LON_MAX:
        x1, y1 = latlon_to_pixel(LAT_MIN, lon)
        x2, y2 = latlon_to_pixel(LAT_MAX, lon)
        cv2.line(img, (x1, y1), (x2, y2), grid_col, 1)
        lon = round(lon + 0.05, 5)

    # Horizontal lines every 0.05° latitude
    lat = LAT_MIN
    while lat <= LAT_MAX:
        x1, y1 = latlon_to_pixel(lat, LON_MIN)
        x2, y2 = latlon_to_pixel(lat, LON_MAX)
        cv2.line(img, (x1, y1), (x2, y2), grid_col, 1)
        lat = round(lat + 0.05, 5)

    # ── 3. Draw ALL bins as small grey dots ───────────────────────────────────
    for _, row in results_df.iterrows():
        px, py = latlon_to_pixel(row["latitude"], row["longitude"])
        cv2.circle(img, (px, py), radius=5, color=(100, 110, 130), thickness=-1)

    # ── 4. Draw route lines ────────────────────────────────────────────────────
    if route:
        # Start from depot
        prev_px, prev_py = latlon_to_pixel(DEPOT_LAT, DEPOT_LON)

        for stop in route:
            curr_px, curr_py = latlon_to_pixel(stop["latitude"], stop["longitude"])
            cv2.line(img,
                     (prev_px, prev_py),
                     (curr_px, curr_py),
                     color=(0, 200, 255),    # Cyan route line
                     thickness=2,
                     lineType=cv2.LINE_AA)
            prev_px, prev_py = curr_px, curr_py

    # ── 5. Draw ACTIVE bins — red, with location label ────────────────────────
    active_ids = set(active_df["bin_id"].tolist()) if not active_df.empty else set()

    # Keep track of where we placed labels to reduce overlap
    drawn_labels = {}    # location_name → pixel (px, py)

    for _, row in results_df.iterrows():
        if row["bin_id"] not in active_ids:
            continue

        px, py = latlon_to_pixel(row["latitude"], row["longitude"])

        # Glow ring
        cv2.circle(img, (px, py), radius=13, color=(0, 0, 100), thickness=-1)
        # Red fill
        cv2.circle(img, (px, py), radius=9, color=(30, 50, 230), thickness=-1)
        # White border
        cv2.circle(img, (px, py), radius=9, color=(255, 255, 255), thickness=1)

        # Draw location label once per unique area (avoid repetition)
        loc = row["location_name"]
        if loc not in drawn_labels:
            drawn_labels[loc] = (px, py)
            label = _short_label(loc)
            cv2.putText(img, label,
                        (px + 12, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (255, 220, 100),   # Warm yellow text
                        1, cv2.LINE_AA)

    # ── 6. Route stop numbers on active bins ──────────────────────────────────
    if route:
        for i, stop in enumerate(route, 1):
            px, py = latlon_to_pixel(stop["latitude"], stop["longitude"])
            num    = str(i)
            (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.putText(img, num,
                        (px - tw // 2, py + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # ── 7. Depot marker (India Gate) ──────────────────────────────────────────
    dpx, dpy = latlon_to_pixel(DEPOT_LAT, DEPOT_LON)
    # Green pentagon-ish star using a diamond shape
    depot_pts = np.array([
        [dpx,      dpy - 13],
        [dpx + 9,  dpy],
        [dpx,      dpy + 13],
        [dpx - 9,  dpy],
    ])
    cv2.fillPoly(img, [depot_pts], color=(0, 210, 80))
    cv2.polylines(img, [depot_pts], isClosed=True,
                  color=(255, 255, 255), thickness=1)
    cv2.putText(img, "India Gate (Depot)",
                (dpx + 12, dpy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 210, 80), 1, cv2.LINE_AA)

    # ── 8. Compass rose (North arrow) ─────────────────────────────────────────
    cx, cy = CANVAS_W - 50, CANVAS_H - 60
    cv2.arrowedLine(img, (cx, cy + 20), (cx, cy - 20),
                    (200, 200, 200), 2, tipLength=0.4)
    cv2.putText(img, "N", (cx - 5, cy - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ── 9. Scale bar (approximate) ────────────────────────────────────────────
    # 0.1 degree longitude ≈ ~8 km at Delhi's latitude
    sb_lon_start = LON_MIN + 0.02
    sb_lon_end   = sb_lon_start + 0.10
    sb_lat       = LAT_MIN + 0.02
    sx1, sy1     = latlon_to_pixel(sb_lat, sb_lon_start)
    sx2, sy2     = latlon_to_pixel(sb_lat, sb_lon_end)

    cv2.line(img, (sx1, sy1), (sx2, sy2), (180, 180, 180), 2)
    cv2.line(img, (sx1, sy1 - 4), (sx1, sy1 + 4), (180, 180, 180), 2)
    cv2.line(img, (sx2, sy2 - 4), (sx2, sy2 + 4), (180, 180, 180), 2)
    cv2.putText(img, "~8 km",
                (sx1 + (sx2 - sx1) // 2 - 20, sy1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    # ── 10. Legend box ────────────────────────────────────────────────────────
    lx, ly = 12, 80
    cv2.rectangle(img, (lx - 4, ly - 4), (lx + 210, ly + 86), (30, 38, 58), -1)
    cv2.rectangle(img, (lx - 4, ly - 4), (lx + 210, ly + 86), (70, 80, 110), 1)

    legend_items = [
        ((100, 110, 130), "Bin — Safe"),
        ((30, 50, 230),   "Bin — Active (overflow risk)"),
        ((0, 200, 255),   "Collection Route"),
        ((0, 210, 80),    "India Gate Depot"),
    ]
    for idx, (col, text) in enumerate(legend_items):
        iy = ly + 10 + idx * 19
        cv2.circle(img, (lx + 7, iy + 3), 5, col, -1)
        cv2.putText(img, text, (lx + 18, iy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 210, 230), 1, cv2.LINE_AA)

    # ── 11. Title & stats ─────────────────────────────────────────────────────
    cv2.putText(img, "WasteWise — Delhi Smart Bin Route Map",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (200, 225, 255), 2, cv2.LINE_AA)

    active_count = len(active_df) if not active_df.empty else 0
    total_km = sum(s["distance_km"] for s in route) if route else 0
    stats = (f"Active bins: {active_count}  |  "
             f"Route stops: {len(route)}  |  "
             f"Total distance: {total_km:.1f} km")
    cv2.putText(img, stats,
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (140, 180, 140), 1, cv2.LINE_AA)

    # ── 12. BGR → RGB for Streamlit ───────────────────────────────────────────
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Quick test
if __name__ == "__main__":
    from data_generator import generate_bin_data
    from model import train_model, predict_overflow
    from route_optimizer import optimize_route

    df = generate_bin_data()
    model, acc, _, _ = train_model(df)
    results_df = predict_overflow(model, df)
    active_df, route, total_km = optimize_route(results_df)

    img = draw_route_map(results_df, active_df, route)
    cv2.imwrite("test_delhi_map.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved test_delhi_map.png  ({total_km:.2f} km route)")
