import folium

# Delhi center coordinates
DELHI_CENTER = [28.6139, 77.2090]

def get_marker_color(fill_level):
    if fill_level > 70:
        return "red"
    elif fill_level >= 40:
        return "orange"
    return "green"

# Delhi map
def draw_route_map(results_df, active_df, route):
    m = folium.Map(
        location=DELHI_CENTER,
        zoom_start=11
    )

    for _, row in results_df.iterrows():
        color = get_marker_color(row["fill_level"])
        popup_text = f"""
        <b>{row['location_name']}</b><br>
        Bin ID: {row['bin_id']}<br>
        Fill Level: {row['fill_level']}%<br>
        Overflow Probability: {row['overflow_prob']}
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_text,
            icon=folium.Icon(color=color, icon="trash")
        ).add_to(m)

    if route:
        route_points = []
        for stop in route:
            route_points.append([
                stop["latitude"],
                stop["longitude"]
            ])

        folium.PolyLine(
            locations=route_points,
            color="blue",
            weight=5,
            opacity=0.8,
            tooltip="Optimized Collection Route"
        ).add_to(m)

    return m
