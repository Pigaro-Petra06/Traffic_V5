import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import os
import pydeck as pdk

# Define your local path
BASE_PATH = r'C:\Kuliah\Bussines Inteligence'

model_path = os.path.join(BASE_PATH, 'traffic_model.h5')
scaler_path = os.path.join(BASE_PATH, 'scaler.gz')

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

st.title("🚦 Toll Jagorawi Monitor")

# 2. Input UI (Moved up so data is available for the map)
st.subheader("Enter Gate Volume")
col1, col2, col3 = st.columns(3)
g1 = col1.number_input("Gate 1", min_value=0, value=0)
g2 = col2.number_input("Gate 2", min_value=0, value=0)
g3 = col3.number_input("Gate 3", min_value=0, value=0)
g4 = col1.number_input("Gate 4", min_value=0, value=0)
g5 = col2.number_input("Gate 5", min_value=0, value=0)
g6 = col3.number_input("Gate 6", min_value=0, value=0)

gate_counts = [g1, g2, g3, g4, g5, g6]

# 1. NEW COORDINATES: Placed in a tight line to look like one Toll Plaza
# Adjusting longitude slightly for each gate to align them side-by-side
center_lat, center_lon = -6.175, 106.827
gate_data = pd.DataFrame([
    {"name": f"Gate {i+1}", "lat": center_lat, "lon": center_lon + (i * 0.0002), "count": gate_counts[i]}
    for i in range(6)
])

# Default color (Grey)
rgb_color = [150, 150, 150, 200]
status_msg = "Awaiting Prediction..."

if st.button("Predict & Update Toll Status", use_container_width=True):
    # 3. ML Prediction Logic
    current_features = np.array([[0, 0, 0, 0, 0, g1, g2, g3, g4, g5, g6]])
    sequence = np.repeat(current_features, 10, axis=0) 
    scaled_data = scaler.transform(sequence).reshape(1, 10, 11)
    
    prediction = model.predict(scaled_data, verbose=0)
    result = np.argmax(prediction)
    
    # 4. Color Mapping
    color_map = {
        0: ([46, 204, 113], "LOW (Safe)"),      
        1: ([243, 156, 18], "MEDIUM (Warning)"), 
        2: ([231, 76, 60], "HIGH (Congested)")   
    }
    rgb_color = color_map[result][0] + [200]
    status_msg = color_map[result][1]

# 5. Build the Map
st.divider()
st.subheader(f"Current Status: {status_msg}")

# View state centered on the new tight gate cluster
view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon + 0.0005, zoom=18, pitch=40)

# Scatterplot for the Gates
layer_points = pdk.Layer(
    "ScatterplotLayer",
    gate_data,
    get_position='[lon, lat]',
    get_color=rgb_color,
    get_radius=10, # Smaller radius because gates are close
    pickable=True
)

# Straight line representing the toll barrier
layer_lines = pdk.Layer(
    "PathLayer",
    [{"path": gate_data[['lon', 'lat']].values.tolist()}],
    get_path="path",
    get_color=rgb_color,
    width_min_pixels=8,
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/dark-v10', # Dark mode looks better for status lights
    initial_view_state=view_state,
    layers=[layer_lines, layer_points],
    tooltip={"text": "{name}\nVehicles: {count}"}
))