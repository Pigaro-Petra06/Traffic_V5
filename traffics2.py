import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os
import pydeck as pdk

# ======================
# CONFIG
# ======================
BASE_PATH = r'C:\Kuliah\Bussines Inteligence'

model_path = os.path.join(BASE_PATH, 'traffic_model.h5')
scaler_path = os.path.join(BASE_PATH, 'scaler.gz')

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception:
    st.error("Model/Scaler not found. Check paths.")
    st.stop()

# ======================
# UI
# ======================
st.title("🚦 Toll Jagorawi Smart Monitor")

st.subheader("Enter Gate Volume")
col1, col2, col3 = st.columns(3)

g1 = col1.number_input("Gate 1", min_value=0, value=10)
g2 = col2.number_input("Gate 2", min_value=0, value=20)
g3 = col3.number_input("Gate 3", min_value=0, value=30)
g4 = col1.number_input("Gate 4", min_value=0, value=15)
g5 = col2.number_input("Gate 5", min_value=0, value=25)
g6 = col3.number_input("Gate 6", min_value=0, value=35)

gate_counts = [g1, g2, g3, g4, g5, g6]

# ======================
# BASE MAP POSITION
# ======================
center_lat, center_lon = -6.175, 106.827

# ======================
# COLOR MAP
# ======================
color_map = {
    0: ([46, 204, 113], "LOW"),
    1: ([243, 156, 18], "MEDIUM"),
    2: ([231, 76, 60], "HIGH")
}

# Default values
gate_results = [0] * 6
gate_colors = [[150, 150, 150, 200]] * 6
status_labels = ["WAITING"] * 6

# ======================
# PREDICTION
# ======================
if st.button("Predict & Update", use_container_width=True):

    gate_results = []

    for g in gate_counts:
        try:
            # ⚠️ IMPORTANT:
            # Adjust feature size if your model expects more inputs
            features = np.array([[0, 0, 0, 0, 0, g]])

            sequence = np.repeat(features, 10, axis=0)

            scaled = scaler.transform(sequence).reshape(1, 10, features.shape[1])

            pred = model.predict(scaled, verbose=0)
            result = np.argmax(pred)

        except:
            # fallback if model mismatch
            if g < 20:
                result = 0
            elif g < 40:
                result = 1
            else:
                result = 2

        gate_results.append(result)

    gate_colors = [color_map[r][0] + [200] for r in gate_results]
    status_labels = [color_map[r][1] for r in gate_results]

# ======================
# GATE POINT DATA
# ======================
gate_data = pd.DataFrame([
    {
        "name": f"Gate {i+1}",
        "lat": center_lat,
        "lon": center_lon + (i * 0.0002),
        "count": gate_counts[i],
        "color": gate_colors[i],
        "status": status_labels[i]
    }
    for i in range(6)
])

# ======================
# LANE (ROAD) DATA
# ======================
lane_data = []

for i in range(6):
    lon = center_lon + (i * 0.0002)

    lane_data.append({
        "path": [
            [lon, center_lat - 0.0006],
            [lon, center_lat]
        ],
        "color": gate_colors[i],
        "name": f"Gate {i+1}",
        "count": gate_counts[i]
    })

# ======================
# VEHICLE QUEUE SIMULATION
# ======================
vehicle_data = []

for i in range(6):
    lon = center_lon + (i * 0.0002)

    # number of cars based on volume
    density = int(gate_counts[i] / 3)

    for j in range(density):
        vehicle_data.append({
            "lon": lon,
            "lat": center_lat - (j * 0.00005),
            "color": gate_colors[i]
        })

vehicle_df = pd.DataFrame(vehicle_data)

# ======================
# MAP VIEW
# ======================
view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon + 0.0005,
    zoom=18,
    pitch=45
)

# ======================
# LAYERS
# ======================

# Lane lines
layer_lanes = pdk.Layer(
    "PathLayer",
    lane_data,
    get_path="path",
    get_color="color",
    width_min_pixels=6,
)

# Gate points
layer_points = pdk.Layer(
    "ScatterplotLayer",
    gate_data,
    get_position='[lon, lat]',
    get_color="color",
    get_radius=15,
    pickable=True
)

# Vehicles
layer_vehicles = pdk.Layer(
    "ScatterplotLayer",
    vehicle_df,
    get_position='[lon, lat]',
    get_color="color",
    get_radius=5,
)

# ======================
# DISPLAY
# ======================
st.divider()
st.subheader("🚦 Gate Status Overview")

for i in range(6):
    st.write(f"Gate {i+1}: {status_labels[i]} ({gate_counts[i]} vehicles)")

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v10',
    initial_view_state=view_state,
    layers=[layer_lanes, layer_vehicles, layer_points],
    tooltip={"text": "{name}\nVehicles: {count}"}
))
