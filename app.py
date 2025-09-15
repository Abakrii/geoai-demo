import streamlit as st
import leafmap.foliumap as leafmap
import numpy as np
from geoai.models import get_model
from geoai.data import load_sample_data

st.set_page_config(layout="wide")
st.title("🌱 GeoAI Demo: Multi-Class Land Cover Segmentation")

st.markdown(
    "Upload a satellite image (GeoTIFF) or use sample data to run multi-class land cover segmentation using GeoAI."
)

# Define class labels and colors for multi-class segmentation
CLASS_LABELS = {
    0: "Vegetation",
    1: "Water", 
    2: "Bare Soil",
    3: "Urban"
}

CLASS_COLORS = {
    0: "#228B22",  # Forest Green for Vegetation
    1: "#4169E1",  # Royal Blue for Water
    2: "#DEB887",  # Burlywood for Bare Soil
    3: "#696969"   # Dim Gray for Urban
}

# Sidebar
with st.sidebar:
    option = st.radio("Choose data:", ["Sample data", "Upload your own"])
    
    if option == "Sample data":
        dataset_options = {
            "Sentinel-2 sample": "sentinel2",
            "Landsat-8 sample": "landsat8", 
            "Drone imagery sample": "drone"
        }
        selected_dataset = st.selectbox(
            "Select Dataset:",
            options=list(dataset_options.keys()),
            index=0
        )
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"])
        selected_dataset = None

# Load data
if uploaded_file is None and option == "Sample data":
    dataset_key = dataset_options[selected_dataset]
    image = load_sample_data(dataset_key)
else:
    image = uploaded_file if uploaded_file is not None else None

if image is not None:
    # Load pretrained multi-class segmentation model
    model = get_model("unet_multiclass")
    mask = model.predict(image)

    st.success("Multi-class segmentation complete ✅")

    # Display class statistics
    if hasattr(mask, 'shape') and len(mask.shape) >= 2:
        unique_classes, counts = np.unique(mask, return_counts=True)
        st.subheader("📊 Segmentation Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Class Distribution:**")
            for class_id, count in zip(unique_classes, counts):
                if class_id in CLASS_LABELS:
                    percentage = (count / mask.size) * 100
                    st.write(f"• {CLASS_LABELS[class_id]}: {percentage:.1f}%")
        
        with col2:
            st.write("**Legend:**")
            for class_id, label in CLASS_LABELS.items():
                color = CLASS_COLORS[class_id]
                st.markdown(f"<span style='color:{color}'>●</span> {label}", unsafe_allow_html=True)

    # Map viewer with multi-class overlay
    m = leafmap.Map(center=[24.45, 54.38], zoom=8)
    m.add_raster(image, colormap="viridis", layer_name="Input Image")
    
    # Add multi-class segmentation overlay
    m.add_raster(mask, colormap="custom", layer_name="Land Cover Classification")
    
    # Add legend to map
    legend_dict = {CLASS_LABELS[i]: CLASS_COLORS[i] for i in CLASS_LABELS.keys()}
    m.add_legend(legend_dict=legend_dict, position="bottomright")
    
    m.to_streamlit(height=600)
