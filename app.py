import streamlit as st
import pandas as pd

from data_generator  import generate_bin_data
from model           import train_model, predict_overflow
from route_optimizer import optimize_route, DEPOT_NAME
from visualization   import draw_route_map
from streamlit_folium import st_folium

st.set_page_config(
    page_title = "WasteWise Delhi",
    layout     = "wide",
)

st.title("🗑️ WasteWise — Delhi Smart Bin Prediction & Route Optimization")
st.markdown(
    """
    **A smart city ML system for Delhi's waste management.**
    Predicts which bins across Delhi's localities are about to overflow,
    then plans the most efficient garbage truck collection route.

    > *Bins are placed across 15 real Delhi areas using accurate GPS coordinates.
    > The truck departs from India Gate and follows a Greedy Nearest-Neighbor route.*
    """
)
st.divider()


for key in ["df", "model", "accuracy", "results_df",
            "active_df", "route", "total_km"]:
    if key not in st.session_state:
        st.session_state[key] = None


with st.sidebar:
    st.header("⚙️ Configuration")

    threshold = st.slider(
        "Overflow Probability Threshold",
        min_value = 0.40,
        max_value = 0.90,
        value     = 0.60,
        step      = 0.05,
        help      = "Bins above this probability are flagged for urgent pickup"
    )

    st.divider()
    st.markdown("**Delhi Areas Covered**")
    areas = [
        "Connaught Place", "Chandni Chowk", "Karol Bagh",
        "Hauz Khas", "Saket", "Dwarka Sector 10", "Lajpat Nagar",
        "Rohini Sector 3", "Janakpuri", "Nehru Place",
        "Pitampura", "Vasant Kunj", "Mayur Vihar Phase 1",
        "Rajouri Garden", "Preet Vihar",
    ]
    for a in areas:
        st.markdown(f"• {a}")

    st.divider()
    st.markdown("**Pipeline Steps**")
    st.markdown("1️⃣ Generate Data")
    st.markdown("2️⃣ Train Model")
    st.markdown("3️⃣ Predict Overflow")
    st.markdown("4️⃣ Optimize Route")
    st.info("Run steps in order.")


# ---------------------------
#  STEP 1 — GENERATE DATA
 # ---------------------------
st.subheader("📊 Step 1: Generate Delhi Bin Dataset")
st.write(
    "Creates a realistic dataset of waste bins spread across 15 Delhi localities. "
    "Each bin has real GPS coordinates, fill level, and overflow history. "
    "Area type (commercial / market / residential) influences fill patterns."
)

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("🔄 Generate Data", use_container_width=True):
        st.session_state["df"] = generate_bin_data()
        # Clear downstream state
        for k in ["model", "accuracy", "results_df", "active_df", "route", "total_km"]:
            st.session_state[k] = None
        total = len(st.session_state["df"])
        st.success(f"✅ Generated {total} bins across 15 Delhi areas!")

if st.session_state["df"] is not None:
    df = st.session_state["df"]

    with col2:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Bins",         len(df))
        m2.metric("Delhi Areas",        df["location_name"].nunique())
        m3.metric("Commercial Bins",    int((df["area_type"] == "commercial").sum()))
        m4.metric("Market Bins",        int((df["area_type"] == "market").sum()))
        m5.metric("Residential Bins",   int((df["area_type"] == "residential").sum()))

    def colour_fill(val):
        """Green for low fill, red for high fill — pure CSS, no matplotlib."""
        if val >= 80:
            return "background-color: #5c1a1a; color: #ffaaaa"
        elif val >= 60:
            return "background-color: #4a3a10; color: #ffd080"
        else:
            return "background-color: #1a3a1a; color: #90ee90"

    display_cols = ["bin_id", "location_name", "latitude", "longitude",
                    "fill_level", "area_type", "past_overflow_count", "overflow"]

    styled = df[display_cols].style.map(colour_fill, subset=["fill_level"])
    st.dataframe(styled, use_container_width=True, height=320)

st.divider()


# ---------------------------
#  STEP 2 — TRAIN MODEL
# ---------------------------
st.subheader("🤖 Step 2: Train ML Model")
st.write(
    "Trains a **Decision Tree Classifier** using fill level, GPS coordinates, "
    "overflow history, and area type as features. 80% of data for training, "
    "20% held back for accuracy testing."
)

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("🧠 Train Model", use_container_width=True):
        if st.session_state["df"] is None:
            st.warning("⚠️ Generate data first (Step 1).")
        else:
            with st.spinner("Training..."):
                model, accuracy, X_test, y_test = train_model(st.session_state["df"])
                st.session_state["model"]    = model
                st.session_state["accuracy"] = accuracy
            st.success(f"✅ Trained! Accuracy: {accuracy * 100:.2f}%")

with col2:
    if st.session_state["accuracy"] is not None:
        acc_pct = st.session_state["accuracy"] * 100
        st.metric("🎯 Test Accuracy", f"{acc_pct:.2f}%")

        if acc_pct >= 85:
            st.success("Excellent — the model learned the Delhi bin overflow pattern well.")
        elif acc_pct >= 70:
            st.info("Good accuracy for a real-world style dataset.")
        else:
            st.warning("Moderate — could improve with more data or feature engineering.")

        st.markdown("""
        **Features used by the model:**
        - `fill_level` — how full the bin is right now
        - `latitude`, `longitude` — real GPS coordinates
        - `past_overflow_count` — overflow history this week
        - `area_type_encoded` — commercial=1, market=2, residential=0
        """)

st.divider()


# ---------------------------
#  STEP 3 — PREDICT OVERFLOW
# ---------------------------
st.subheader("🔮 Step 3: Predict Overflow Probability")
st.write(
    "Runs the trained model on every Delhi bin and scores each with an "
    f"overflow probability. Bins above **{threshold:.0%}** are flagged for collection."
)

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("🔍 Predict Overflow", use_container_width=True):
        if st.session_state["model"] is None:
            st.warning("⚠️ Train the model first (Step 2).")
        else:
            results_df = predict_overflow(
                st.session_state["model"],
                st.session_state["df"]
            )
            st.session_state["results_df"] = results_df
            flagged = int((results_df["overflow_prob"] > threshold).sum())
            st.success(f"✅ Done! {flagged} bins flagged across Delhi.")

if st.session_state["results_df"] is not None:
    results_df = st.session_state["results_df"]
    flagged    = int((results_df["overflow_prob"] > threshold).sum())

    with col2:
        m1, m2, m3 = st.columns(3)
        m1.metric("🔴 Flagged Bins",      flagged)
        m2.metric("🟢 Safe Bins",         len(results_df) - flagged)
        m3.metric("📈 Avg Overflow Prob", f"{results_df['overflow_prob'].mean():.1%}")

    # Colour rows: red tint if flagged, green tint if safe
    def highlight_row(row):
        if row["overflow_prob"] > threshold:
            return ["background-color: #3d1010"] * len(row)
        return ["background-color: #0f2a0f"] * len(row)

    disp_cols = ["bin_id", "location_name", "area_type",
                 "fill_level", "past_overflow_count", "overflow_prob", "predicted"]

    styled_res = (
        results_df[disp_cols]
        .style
        .apply(highlight_row, axis=1)
        .format({"overflow_prob": "{:.3f}"})
    )
    st.dataframe(styled_res, use_container_width=True, height=360)
    st.caption(f"🔴 Red rows = overflow probability > {threshold} (urgent pickup needed)")

st.divider()


# ---------------------------
#  STEP 4 — OPTIMIZE ROUTE
# ---------------------------

st.subheader("🗺️ Step 4: Optimize Collection Route (Delhi)")
st.write(
    f"Plans the garbage truck route starting from **{DEPOT_NAME}**. "
    "Uses Greedy Nearest Neighbor with real Haversine (GPS) distances. "
    "Route distances are shown in **kilometres**."
)

col1, col2 = st.columns([1, 4])

with col1:
    if st.button("🚛 Optimize Route", use_container_width=True):
        if st.session_state["results_df"] is None:
            st.warning("⚠️ Run predictions first (Step 3).")
        else:
            with st.spinner("Planning route across Delhi..."):
                active_df, route, total_km = optimize_route(
                    st.session_state["results_df"],
                    threshold=threshold
                )
                st.session_state["active_df"] = active_df
                st.session_state["route"]     = route
                st.session_state["total_km"]  = total_km
            st.success(
                f"✅ Route planned! {len(route)} stops, "
                f"{total_km:.2f} km total."
            )

if st.session_state["route"] is not None:
    route      = st.session_state["route"]
    active_df  = st.session_state["active_df"]
    total_km   = st.session_state["total_km"]
    results_df = st.session_state["results_df"]

    with col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📍 Total Stops",    len(route))
        m2.metric("📏 Distance",       f"{total_km:.2f} km")
        m3.metric("🏁 Start Point",    "India Gate")
        m4.metric("⚡ Algorithm",      "Greedy NN")

#map
    st.markdown("#### 🗺️ Delhi Bin Route Map")

    folium_map = draw_route_map(
        results_df,
        active_df,
        route
    )

    st_folium(
        folium_map,
        width=1200,
        height=600
    )

# Route 
    st.markdown("#### 📋 Route Sequence")

    if route:
        route_df = pd.DataFrame(route)
        route_df.index = route_df.index + 1
        route_df.index.name = "Stop #"
        route_df = route_df.rename(columns={
            "bin_id"        : "Bin ID",
            "location_name" : "Delhi Area",
            "latitude"      : "Latitude",
            "longitude"     : "Longitude",
            "distance_km"   : "Distance from Prev (km)",
        })
        st.dataframe(route_df, use_container_width=True)
    else:
        st.info("No active bins at this threshold — try lowering it in the sidebar.")

    #  Active bins detail 
    if not active_df.empty:
        with st.expander("📦 Active Bins Detail — All Flagged Delhi Bins"):
            disp = active_df[[
                "bin_id", "location_name", "area_type",
                "fill_level", "past_overflow_count", "overflow_prob"
            ]].style.format({"overflow_prob": "{:.3f}"})
            st.dataframe(disp, use_container_width=True)

        # Area breakdown pie-style summary
        st.markdown("#### 📊 Active Bins by Delhi Area")
        area_counts = (
            active_df.groupby("location_name")
            .size()
            .reset_index(name="Active Bins")
            .sort_values("Active Bins", ascending=False)
        )
        st.bar_chart(area_counts.set_index("location_name"))

st.divider()

# Footer 
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.82em; padding:8px 0'>
    WasteWise v2 — Delhi &nbsp;|&nbsp;
    Python · Scikit-Learn · Folium · Streamlit &nbsp;|&nbsp;
    15 Delhi localities
</div>
""", unsafe_allow_html=True)
