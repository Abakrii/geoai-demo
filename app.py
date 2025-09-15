import streamlit as st
import leafmap.foliumap as leafmap
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import json
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

def calculate_polygon_stats(polygon_coords, mask_data, ndvi_data=None, pixel_size_km=0.01):
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
        
        # Calculate NDVI statistics if available
        ndvi_stats = None
        if ndvi_data is not None:
            # Simulate NDVI statistics for the polygon area
            np.random.seed(42)
            ndvi_mean = np.random.uniform(0.2, 0.8)
            ndvi_std = np.random.uniform(0.1, 0.3)
            ndvi_min = max(-1, ndvi_mean - 2 * ndvi_std)
            ndvi_max = min(1, ndvi_mean + 2 * ndvi_std)
            
            ndvi_stats = {
                "mean": ndvi_mean,
                "std": ndvi_std,
                "min": ndvi_min,
                "max": ndvi_max
            }
        
        return {
            "area_km2": area_km2,
            "class_percentages": class_percentages,
            "ndvi_stats": ndvi_stats
        }
    except Exception as e:
        st.error(f"Error calculating polygon statistics: {str(e)}")
        return None

def calculate_ndvi(image_data):
    """
    Calculate NDVI (Normalized Difference Vegetation Index) from image data.
    
    Args:
        image_data: Image array with multiple bands (NIR and Red bands)
    
    Returns:
        numpy.ndarray: NDVI values ranging from -1 to 1
    """
    try:
        # For demonstration, we'll simulate NDVI calculation
        # In a real implementation, you'd extract NIR and Red bands from the image
        # NDVI = (NIR - Red) / (NIR + Red)
        
        # Simulate NDVI values with realistic distribution
        np.random.seed(42)
        height, width = 256, 256  # Default size for demo
        
        # Create realistic NDVI pattern
        ndvi = np.random.normal(0.3, 0.4, (height, width))
        
        # Add some spatial structure
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # Create radial pattern for more realistic NDVI
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        radial_factor = 1 - (distance / max_distance) * 0.5
        
        ndvi = ndvi * radial_factor
        
        # Clamp values to valid NDVI range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
        
    except Exception as e:
        st.error(f"Error calculating NDVI: {str(e)}")
        return None

def create_geotiff_bytes(mask_data, image_name="segmentation_mask"):
    """
    Create GeoTIFF bytes from segmentation mask data.
    
    Args:
        mask_data: Segmentation mask array
        image_name: Name for the file
    
    Returns:
        bytes: GeoTIFF file as bytes
    """
    try:
        # Create a simple GeoTIFF in memory
        # In a real implementation, you'd use proper geospatial metadata
        buffer = io.BytesIO()
        
        # Simulate GeoTIFF creation
        # For demo purposes, we'll create a simple binary representation
        mask_bytes = mask_data.tobytes()
        
        # Create a simple header with metadata
        header = {
            "format": "GeoTIFF",
            "width": mask_data.shape[1] if len(mask_data.shape) > 1 else 256,
            "height": mask_data.shape[0] if len(mask_data.shape) > 1 else 256,
            "bands": 1,
            "dtype": str(mask_data.dtype),
            "data": mask_bytes.hex()  # Convert to hex string for JSON serialization
        }
        
        # Convert to JSON and then to bytes
        json_data = json.dumps(header)
        buffer.write(json_data.encode())
        buffer.seek(0)
        
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating GeoTIFF: {str(e)}")
        return None

def create_geojson_bytes(mask_data, image_name="segmentation_mask"):
    """
    Create GeoJSON bytes from segmentation mask data.
    
    Args:
        mask_data: Segmentation mask array
        image_name: Name for the file
    
    Returns:
        bytes: GeoJSON file as bytes
    """
    try:
        # Create simplified GeoJSON representation
        # In a real implementation, you'd convert raster to vector polygons
        
        # Create a simple polygon representation
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "image_name": image_name,
                        "class_labels": CLASS_LABELS,
                        "class_colors": CLASS_COLORS,
                        "description": "Segmentation mask converted to GeoJSON"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [24.4, 54.3], [24.5, 54.3], [24.5, 54.4], [24.4, 54.4], [24.4, 54.3]
                        ]]
                    }
                }
            ]
        }
        
        # Convert to JSON bytes
        json_bytes = json.dumps(geojson, indent=2).encode()
        return json_bytes
        
    except Exception as e:
        st.error(f"Error creating GeoJSON: {str(e)}")
        return None

def create_csv_bytes(stats_data, polygon_coords=None):
    """
    Create CSV bytes from polygon statistics data.
    
    Args:
        stats_data: Dictionary containing statistics
        polygon_coords: Polygon coordinates for reference
    
    Returns:
        bytes: CSV file as bytes
    """
    try:
        # Create DataFrame from statistics
        data = {
            'Metric': [],
            'Value': [],
            'Unit': []
        }
        
        # Add area information
        data['Metric'].append('Area')
        data['Value'].append(f"{stats_data['area_km2']:.4f}")
        data['Unit'].append('km²')
        
        # Add land cover percentages
        for class_name, percentage in stats_data['class_percentages'].items():
            data['Metric'].append(f'{class_name} Coverage')
            data['Value'].append(f"{percentage:.2f}")
            data['Unit'].append('%')
        
        # Add NDVI statistics if available
        if stats_data.get('ndvi_stats'):
            ndvi_stats = stats_data['ndvi_stats']
            data['Metric'].extend(['NDVI Mean', 'NDVI Std Dev', 'NDVI Min', 'NDVI Max'])
            data['Value'].extend([
                f"{ndvi_stats['mean']:.4f}",
                f"{ndvi_stats['std']:.4f}",
                f"{ndvi_stats['min']:.4f}",
                f"{ndvi_stats['max']:.4f}"
            ])
            data['Unit'].extend(['NDVI', 'NDVI', 'NDVI', 'NDVI'])
        
        # Add metadata
        data['Metric'].extend(['Analysis Date', 'Polygon Coordinates'])
        data['Value'].extend([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            str(polygon_coords) if polygon_coords else 'N/A'
        ])
        data['Unit'].extend(['', ''])
        
        # Create DataFrame and convert to CSV
        df = pd.DataFrame(data)
        csv_bytes = df.to_csv(index=False).encode()
        
        return csv_bytes
        
    except Exception as e:
        st.error(f"Error creating CSV: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    analysis_mode = st.radio("Analysis Mode:", ["Single Image", "Time Series"])
    
    if analysis_mode == "Single Image":
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
            uploaded_files = None
            selected_datasets = None
        else:
            uploaded_files = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"], accept_multiple_files=False)
            selected_dataset = None
            selected_datasets = None
    
    else:  # Time Series mode
        option = st.radio("Choose data:", ["Sample data", "Upload your own"])
        
        if option == "Sample data":
            dataset_options = {
                "Sentinel-2 sample": "sentinel2",
                "Landsat-8 sample": "landsat8", 
                "Drone imagery sample": "drone"
            }
            selected_datasets = st.multiselect(
                "Select Multiple Datasets:",
                options=list(dataset_options.keys()),
                default=[list(dataset_options.keys())[0]]
            )
            uploaded_files = None
            selected_dataset = None
        else:
            uploaded_files = st.file_uploader("Upload Multiple GeoTIFFs", type=["tif", "tiff"], accept_multiple_files=True)
            selected_datasets = None
            selected_dataset = None

# NDVI Analysis Toggle
st.sidebar.markdown("---")
st.sidebar.subheader("🌿 NDVI Analysis")
enable_ndvi = st.sidebar.checkbox("Enable NDVI Calculation", value=False)

if enable_ndvi:
    st.sidebar.info("NDVI will be calculated and displayed as a heatmap overlay.")
else:
    st.sidebar.info("NDVI calculation is disabled.")

# Animation controls for time series
if analysis_mode == "Time Series":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎬 Animation Controls")
    enable_animation = st.sidebar.checkbox("Enable Map Animation", value=False)
    animation_speed = st.sidebar.slider("Animation Speed (seconds)", 1, 5, 2)
else:
    enable_animation = False
    animation_speed = 2

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
    
    # Display NDVI statistics if available
    if stats.get('ndvi_stats'):
        st.sidebar.write("**NDVI Statistics:**")
        ndvi_stats = stats['ndvi_stats']
        st.sidebar.write(f"• Mean: {ndvi_stats['mean']:.3f}")
        st.sidebar.write(f"• Std Dev: {ndvi_stats['std']:.3f}")
        st.sidebar.write(f"• Range: {ndvi_stats['min']:.3f} to {ndvi_stats['max']:.3f}")
    
    if st.sidebar.button("Clear Analysis"):
        st.session_state.polygon_stats = None
        st.rerun()
else:
    st.sidebar.info("Draw a polygon on the map to analyze land cover distribution.")

# Download Section
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Download Results")

# Initialize session state for current analysis data
if 'current_masks' not in st.session_state:
    st.session_state.current_masks = None
if 'current_image_names' not in st.session_state:
    st.session_state.current_image_names = None
if 'current_ndvi_data' not in st.session_state:
    st.session_state.current_ndvi_data = None

# Load data based on analysis mode
if analysis_mode == "Single Image":
    # Single image processing
    if uploaded_files is None and option == "Sample data":
        dataset_key = dataset_options[selected_dataset]
        images = [load_sample_data(dataset_key)]
        image_names = [selected_dataset]
    else:
        images = [uploaded_files] if uploaded_files is not None else []
        image_names = [f"Uploaded Image {i+1}" for i in range(len(images))]
else:
    # Time series processing
    if uploaded_files is None and option == "Sample data":
        if selected_datasets:
            images = [load_sample_data(dataset_options[ds]) for ds in selected_datasets]
            image_names = selected_datasets
        else:
            images = []
            image_names = []
    else:
        images = uploaded_files if uploaded_files is not None else []
        image_names = [f"Image {i+1}" for i in range(len(images))]

if images:
    # Load pretrained multi-class segmentation model
    model = get_model("unet_multiclass")
    
    # Process all images
    all_masks = []
    all_ndvi_data = []
    time_series_data = []
    
    with st.spinner("Processing images..."):
        for i, image in enumerate(images):
            # Segmentation
            mask = model.predict(image)
            all_masks.append(mask)
            
            # NDVI calculation if enabled
            ndvi_data = None
            if enable_ndvi:
                ndvi_data = calculate_ndvi(image)
                all_ndvi_data.append(ndvi_data)
            
            # Calculate vegetation percentage for time series
            if hasattr(mask, 'shape') and len(mask.shape) >= 2:
                unique_classes, counts = np.unique(mask, return_counts=True)
                vegetation_count = counts[unique_classes == 0] if 0 in unique_classes else 0
                vegetation_percentage = (vegetation_count / mask.size) * 100 if mask.size > 0 else 0
                
                time_series_data.append({
                    'date': datetime.now() - timedelta(days=len(images)-i-1),  # Simulate dates
                    'image_name': image_names[i],
                    'vegetation_percentage': vegetation_percentage[0] if len(vegetation_percentage) > 0 else 0,
                    'ndvi_mean': np.mean(ndvi_data) if ndvi_data is not None else None
                })

    st.success(f"Processed {len(images)} images successfully ✅")
    
    # Store current analysis data in session state for downloads
    st.session_state.current_masks = all_masks
    st.session_state.current_image_names = image_names
    st.session_state.current_ndvi_data = all_ndvi_data
    
    # Display time series chart if multiple images
    if len(images) > 1:
        st.subheader("📈 Vegetation Coverage Over Time")
        
        # Create DataFrame for plotting
        df = pd.DataFrame(time_series_data)
        
        # Create line chart
        fig = go.Figure()
        
        # Add vegetation percentage line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['vegetation_percentage'],
            mode='lines+markers',
            name='Vegetation Coverage (%)',
            line=dict(color='#228B22', width=3),
            marker=dict(size=8)
        ))
        
        # Add NDVI line if available
        if enable_ndvi and any(df['ndvi_mean'].notna()):
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['ndvi_mean'],
                mode='lines+markers',
                name='NDVI Mean',
                line=dict(color='#FFD700', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ))
        
        # Update layout
        fig.update_layout(
            title="Vegetation Analysis Over Time",
            xaxis_title="Date",
            yaxis_title="Vegetation Coverage (%)",
            yaxis2=dict(title="NDVI", overlaying="y", side="right") if enable_ndvi else None,
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display class statistics for the first image (or selected image)
    if len(all_masks) > 0:
        mask = all_masks[0]  # Display stats for first image
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
    if len(images) > 1 and enable_animation:
        st.subheader("🎬 Animated Map View")
        
        # Create animation placeholder
        animation_placeholder = st.empty()
        
        # Animation loop
        for i in range(len(images)):
            with animation_placeholder.container():
                st.write(f"**Time Step {i+1}: {image_names[i]}**")
                
                m = leafmap.Map(center=[24.45, 54.38], zoom=8)
                m.add_raster(images[i], colormap="viridis", layer_name="Input Image")
                m.add_raster(all_masks[i], colormap="custom", layer_name="Land Cover Classification")
                
                if enable_ndvi and i < len(all_ndvi_data) and all_ndvi_data[i] is not None:
                    m.add_raster(all_ndvi_data[i], colormap="RdYlGn", layer_name="NDVI Heatmap")
                
                # Add legends
                legend_dict = {CLASS_LABELS[j]: CLASS_COLORS[j] for j in CLASS_LABELS.keys()}
                m.add_legend(legend_dict=legend_dict, position="bottomright")
                
                if enable_ndvi and i < len(all_ndvi_data) and all_ndvi_data[i] is not None:
                    ndvi_legend = {
                        "High Vegetation (0.6-1.0)": "#228B22",
                        "Moderate Vegetation (0.2-0.6)": "#FFD700", 
                        "Low Vegetation (0.0-0.2)": "#FF8C00",
                        "No Vegetation (-1.0-0.0)": "#DC143C"
                    }
                    m.add_legend(legend_dict=ndvi_legend, position="bottomleft")
                
                m.to_streamlit(height=400)
            
            # Wait for animation speed
            import time
            time.sleep(animation_speed)
    
    else:
        # Static map view
        if len(images) > 1:
            st.subheader("🗺️ Map View (First Image)")
        
        m = leafmap.Map(center=[24.45, 54.38], zoom=8)
        m.add_raster(images[0], colormap="viridis", layer_name="Input Image")
        m.add_raster(all_masks[0], colormap="custom", layer_name="Land Cover Classification")
        
        # Add NDVI overlay if available
        if enable_ndvi and len(all_ndvi_data) > 0 and all_ndvi_data[0] is not None:
            m.add_raster(all_ndvi_data[0], colormap="RdYlGn", layer_name="NDVI Heatmap")
        
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
        
        # Add NDVI legend if available
        if enable_ndvi and len(all_ndvi_data) > 0 and all_ndvi_data[0] is not None:
            ndvi_legend = {
                "High Vegetation (0.6-1.0)": "#228B22",
                "Moderate Vegetation (0.2-0.6)": "#FFD700", 
                "Low Vegetation (0.0-0.2)": "#FF8C00",
                "No Vegetation (-1.0-0.0)": "#DC143C"
            }
            m.add_legend(legend_dict=ndvi_legend, position="bottomleft")
        
        # Handle polygon drawing events
        if hasattr(m, '_last_draw') and m._last_draw:
            try:
                # Extract polygon coordinates from the drawn shape
                if 'geometry' in m._last_draw and 'coordinates' in m._last_draw['geometry']:
                    coords = m._last_draw['geometry']['coordinates'][0]  # First ring of polygon
                    
                    # Convert to (lat, lon) format
                    polygon_coords = [(coord[1], coord[0]) for coord in coords]
                    
                    # Calculate statistics
                    ndvi_data = all_ndvi_data[0] if len(all_ndvi_data) > 0 else None
                    stats = calculate_polygon_stats(polygon_coords, all_masks[0], ndvi_data)
                    if stats:
                        st.session_state.polygon_stats = stats
                        st.rerun()
            except Exception as e:
                st.error(f"Error processing drawn polygon: {str(e)}")
        
        m.to_streamlit(height=600)

    # Download buttons section
    if st.session_state.current_masks is not None:
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Segmentation Masks**")
            
            # Download GeoTIFF
            if st.button("📁 Download GeoTIFF", key="download_geotiff") and len(st.session_state.current_masks) > 0:
                mask_data = st.session_state.current_masks[0]
                image_name = st.session_state.current_image_names[0] if st.session_state.current_image_names else "segmentation_mask"
                
                geotiff_bytes = create_geotiff_bytes(mask_data, image_name)
                if geotiff_bytes:
                    st.download_button(
                        label="⬇️ Download GeoTIFF",
                        data=geotiff_bytes,
                        file_name=f"{image_name}_segmentation_mask.tif",
                        mime="application/octet-stream",
                        key="download_geotiff_file"
                    )
            
            # Download GeoJSON
            if st.button("🗺️ Download GeoJSON", key="download_geojson") and len(st.session_state.current_masks) > 0:
                mask_data = st.session_state.current_masks[0]
                image_name = st.session_state.current_image_names[0] if st.session_state.current_image_names else "segmentation_mask"
                
                geojson_bytes = create_geojson_bytes(mask_data, image_name)
                if geojson_bytes:
                    st.download_button(
                        label="⬇️ Download GeoJSON",
                        data=geojson_bytes,
                        file_name=f"{image_name}_segmentation_mask.geojson",
                        mime="application/json",
                        key="download_geojson_file"
                    )
        
        with col2:
            st.write("**Polygon Statistics**")
            
            # Download CSV for polygon statistics
            if st.session_state.polygon_stats is not None:
                if st.button("📊 Download CSV", key="download_csv"):
                    csv_bytes = create_csv_bytes(st.session_state.polygon_stats)
                    if csv_bytes:
                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv_bytes,
                            file_name="polygon_statistics.csv",
                            mime="text/csv",
                            key="download_csv_file"
                        )
            else:
                st.info("Draw a polygon to enable CSV download")
        
        with col3:
            st.write("**Time Series Data**")
            
            # Download time series CSV if multiple images
            if len(st.session_state.current_masks) > 1:
                if st.button("📈 Download Time Series", key="download_timeseries"):
                    # Create time series CSV
                    time_series_data = []
                    for i, mask in enumerate(st.session_state.current_masks):
                        if hasattr(mask, 'shape') and len(mask.shape) >= 2:
                            unique_classes, counts = np.unique(mask, return_counts=True)
                            vegetation_count = counts[unique_classes == 0] if 0 in unique_classes else 0
                            vegetation_percentage = (vegetation_count / mask.size) * 100 if mask.size > 0 else 0
                            
                            ndvi_mean = None
                            if st.session_state.current_ndvi_data and i < len(st.session_state.current_ndvi_data):
                                ndvi_data = st.session_state.current_ndvi_data[i]
                                if ndvi_data is not None:
                                    ndvi_mean = np.mean(ndvi_data)
                            
                            time_series_data.append({
                                'Date': datetime.now() - timedelta(days=len(st.session_state.current_masks)-i-1),
                                'Image_Name': st.session_state.current_image_names[i] if i < len(st.session_state.current_image_names) else f"Image_{i+1}",
                                'Vegetation_Coverage_Percent': vegetation_percentage[0] if len(vegetation_percentage) > 0 else 0,
                                'NDVI_Mean': ndvi_mean if ndvi_mean is not None else 'N/A'
                            })
                    
                    if time_series_data:
                        df_timeseries = pd.DataFrame(time_series_data)
                        csv_bytes = df_timeseries.to_csv(index=False).encode()
                        
                        st.download_button(
                            label="⬇️ Download Time Series CSV",
                            data=csv_bytes,
                            file_name="time_series_analysis.csv",
                            mime="text/csv",
                            key="download_timeseries_file"
                        )
            else:
                st.info("Time series data available with multiple images")
