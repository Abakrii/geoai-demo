import streamlit as st
import leafmap.foliumap as leafmap
from geoai.models import get_model
from geoai.data import load_sample_data

st.set_page_config(layout="wide")
st.title("🌱 GeoAI Demo: Vegetation Segmentation")

st.markdown(
    "Upload a satellite image (GeoTIFF) or use sample data to run vegetation segmentation using GeoAI."
)

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
    # Load pretrained segmentation model (simplified)
    model = get_model("unet")
    mask = model.predict(image)

    st.success("Segmentation complete ✅")

    # Map viewer
    m = leafmap.Map(center=[24.45, 54.38], zoom=8)
    m.add_raster(image, colormap="viridis", layer_name="Input Image")
    m.add_raster(mask, colormap="Greens", layer_name="Vegetation Mask")
    m.to_streamlit(height=600)
