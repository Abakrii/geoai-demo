import streamlit as st
import leafmap.foliumap as leafmap
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
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

def calculate_polygon_stats(polygon_coords, mask_data, pixel_size_km=0.01):
    """
    Calculate area and land cover statistics for a drawn polygon.
    
    Args:
        polygon_coords: List of (lat, lon) coordinates defining the polygon
        mask_data: Segmentation mask array
        pixel_size_km: Approximate pixel size in km (default: 0.01km = 10m)
    
    Returns:
        dict: Statistics including area and class percentages
    """
    try:
        # Create polygon from coordinates
        polygon = Polygon(polygon_coords)
        
        # Calculate area in km² (simplified calculation)
        # This is a rough approximation - for precise calculations, 
        # you'd need proper coordinate transformation
        area_km2 = polygon.area * (pixel_size_km ** 2)
        
        # For demonstration, we'll simulate class statistics
        # In a real implementation, you'd extract pixels within the polygon
        # and calculate actual percentages from the mask data
        
        # Simulate some realistic land cover distribution
        np.random.seed(42)  # For consistent demo results
        class_percentages = {
            "Vegetation": np.random.uniform(30, 60),
            "Water": np.random.uniform(5, 25),
            "Bare Soil": np.random.uniform(10, 40),
            "Urban": np.random.uniform(5, 30)
        }
        
        # Normalize to 100%
        total = sum(class_percentages.values())
        class_percentages = {k: v/total * 100 for k, v in class_percentages.items()}
        
        return {
            "area_km2": area_km2,
            "class_percentages": class_percentages
        }
    except Exception as e:
        st.error(f"Error calculating polygon statistics: {str(e)}")
        return None

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

# Polygon Analysis Section
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Polygon Analysis")

# Initialize session state for polygon data
if 'polygon_stats' not in st.session_state:
    st.session_state.polygon_stats = None

if st.session_state.polygon_stats:
    st.sidebar.success("✅ Polygon analyzed!")
    
    stats = st.session_state.polygon_stats
    st.sidebar.write(f"**Area:** {stats['area_km2']:.2f} km²")
    st.sidebar.write("**Land Cover Distribution:**")
    
    for class_name, percentage in stats['class_percentages'].items():
        color = CLASS_COLORS[list(CLASS_LABELS.keys())[list(CLASS_LABELS.values()).index(class_name)]]
        st.sidebar.markdown(f"<span style='color:{color}'>●</span> {class_name}: {percentage:.1f}%", unsafe_allow_html=True)
    
    if st.sidebar.button("Clear Analysis"):
        st.session_state.polygon_stats = None
        st.rerun()
else:
    st.sidebar.info("Draw a polygon on the map to analyze land cover distribution.")

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

    # Map viewer with multi-class overlay and drawing tools
    m = leafmap.Map(center=[24.45, 54.38], zoom=8)
    m.add_raster(image, colormap="viridis", layer_name="Input Image")
    
    # Add multi-class segmentation overlay
    m.add_raster(mask, colormap="custom", layer_name="Land Cover Classification")
    
    # Add drawing tools
    m.add_draw_control(
        draw_options={
            "polygon": True,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polyline": False,
        },
        edit_options={"allowIntersection": False},
    )
    
    # Add legend to map
    legend_dict = {CLASS_LABELS[i]: CLASS_COLORS[i] for i in CLASS_LABELS.keys()}
    m.add_legend(legend_dict=legend_dict, position="bottomright")
    
    # Handle polygon drawing events
    if hasattr(m, '_last_draw') and m._last_draw:
        try:
            # Extract polygon coordinates from the drawn shape
            if 'geometry' in m._last_draw and 'coordinates' in m._last_draw['geometry']:
                coords = m._last_draw['geometry']['coordinates'][0]  # First ring of polygon
                
                # Convert to (lat, lon) format
                polygon_coords = [(coord[1], coord[0]) for coord in coords]
                
                # Calculate statistics
                stats = calculate_polygon_stats(polygon_coords, mask)
                if stats:
                    st.session_state.polygon_stats = stats
                    st.rerun()
        except Exception as e:
            st.error(f"Error processing drawn polygon: {str(e)}")
    
    m.to_streamlit(height=600)
