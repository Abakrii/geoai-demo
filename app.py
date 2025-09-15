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
    if option == "Upload your own":
        uploaded_file = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"])
    else:
        uploaded_file = None

# Load data
if uploaded_file is None and option == "Sample data":
    image = load_sample_data("sentinel2")
else:
    if uploaded_file is not None:
        image = uploaded_file
    else:
        image = None

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
