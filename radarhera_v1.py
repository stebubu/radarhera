import streamlit as st
import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_origin
import folium
from folium import raster_layers
import tempfile
import requests
from datetime import datetime, timedelta
import os
import http.server
import socketserver
import threading
import time

# Authentication and request parameters
auth_url = "https://api.hypermeteo.com/auth-b2b/authenticate"
body = {
    "username": "gecosistema",
    "password": "Saferplaces2023!"
}
response = requests.post(auth_url, json=body).json()
token = response['token']
headers = {"Authorization": f"Bearer {token}"}

base_url = "https://api.hypermeteo.com/b2b-binary/ogc/geoserver/wcs"
service = "WCS"
request_type = "GetCoverage"
version = "2.0.0"
coverage_id = "RADAR_HERA_150M_5MIN__rainrate"
format_type = "application/x-netcdf"
subset_lat = "Lat(43.8,44.2)"
subset_lon = "Long(12.4,12.9)"

# Coordinates and cell size
lon_min = 12.4
lon_max = 12.9
lat_min = 43.8
lat_max = 44.2
cell_size_lon = 0.001873207092285156293
cell_size_lat = -0.00134747044569781039

# Streamlit app layout
st.title("Rain Rate Mapping")
st.sidebar.title("Settings")

# Date selection
selected_date = st.sidebar.date_input("Select date", datetime.utcnow().date())

# Hour selection
selected_hour = st.sidebar.selectbox("Select hour of the day", options=range(24), index=datetime.utcnow().hour)

# Minute selection in 5-minute increments
selected_minute = st.sidebar.select_slider("Select minute of the hour", options=list(range(0, 60, 5)), value=(datetime.utcnow().minute // 5) * 5)

# Combine selected date, hour, and minute into a datetime object
selected_time = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour, minutes=selected_minute)

# Select cumulative interval
cumulative_options = {
    "No Cumulative": timedelta(minutes=5),
    "Last 30 min": timedelta(minutes=30),
    "Last 1 hour": timedelta(hours=1),
    "Last 3 hours": timedelta(hours=3)
}
cumulative_interval = st.sidebar.selectbox("Cumulative interval", options=list(cumulative_options.keys()))

# Calculate start and end time
end_time = selected_time
start_time = end_time - cumulative_options[cumulative_interval]

# Fetch data from API
def fetch_rain_data(start_time, end_time):
    current_time = start_time
    rain_data = []
    temp_files = []  # List to keep track of temporary files for later cleanup
    
    while current_time <= end_time:
        subset_time = f'time("{current_time.isoformat(timespec="milliseconds")}Z")'

        params = {
            "request": request_type,
            "service": service,
            "version": version,
            "coverageId": coverage_id,
            "format": format_type,
            "subset": [subset_lon, subset_lat, subset_time]
        }

        response = requests.get(base_url, headers=headers, params=params)
        
        if response.status_code == 200:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.nc')
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
            tmp_file.close()
            temp_files.append(tmp_file_path)

            try:
                # Open the dataset from the temporary file
                ds = xr.open_dataset(tmp_file_path, engine='netcdf4')
                rain_data.append(ds)
            except Exception as e:
                st.error(f"Failed to open dataset: {e}")
        else:
            st.error(f"Error fetching data for {current_time}: {response.text}")
            break
        
        current_time += timedelta(minutes=5)
    
    # Clean up temporary files
    for file_path in temp_files:
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"Failed to remove temporary file: {file_path}. Error: {e}")
    
    return rain_data

# Convert NetCDF to GeoTIFF
def fetch_rain_data_as_geotiff(rain_data):
    if rain_data and len(rain_data) > 0:
        combined_data = rain_data[0]  # Use the single time step data

        # Extract lat, lon, and rainrate
        lat = combined_data.coords['lat'].values
        lon = combined_data.coords['lon'].values
        rainrate = combined_data['rainrate'].squeeze().values  # Remove the single time dimension

        # Print shapes for debugging
        st.write(f"Latitude shape: {lat.shape}")
        st.write(f"Longitude shape: {lon.shape}")
        st.write(f"Rainrate shape: {rainrate.shape}")

        # Use np.meshgrid to align lat and lon with rainrate
        lon, lat = np.meshgrid(lon, lat)

        # Re-check shapes after meshgrid
        st.write(f"After meshgrid - Latitude shape: {lat.shape}")
        st.write(f"After meshgrid - Longitude shape: {lon.shape}")

        if lat.shape == lon.shape == rainrate.shape:
            # Create the GeoTIFF using rasterio
            transform = from_origin(lon_min, lat_max, cell_size_lon, abs(cell_size_lat))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                with rasterio.open(
                    tmp_file.name,
                    'w',
                    driver='GTiff',
                    height=rainrate.shape[0],
                    width=rainrate.shape[1],
                    count=1,
                    dtype=rainrate.dtype,
                    crs='EPSG:4326',
                    transform=transform,
                ) as dst:
                    dst.write(rainrate, 1)
                geotiff_path = tmp_file.name
                return geotiff_path
        else:
            st.error("Mismatch in array dimensions: lat, lon, and rainrate must have the same shape.")
            return None
    else:
        st.warning("No data available for the selected time and cumulative interval.")
        return None

# Start a temporary HTTP server to serve the GeoTIFF
def serve_geotiff(geotiff_path):
    handler = http.server.SimpleHTTPRequestHandler
    os.chdir(os.path.dirname(geotiff_path))
    
    port = 8000  # You can choose any available port
    httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
    
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    
    # Return the URL of the served file
    geotiff_url = f"http://localhost:{port}/{os.path.basename(geotiff_path)}"
    
    return geotiff_url, httpd

# Map the GeoTIFF using folium
def map_geotiff(geotiff_url):
    try:
        m = folium.Map(location=[(lat_max + lat_min) / 2, (lon_max + lon_min) / 2], zoom_start=10)
        raster = raster_layers.ImageOverlay(
            image=geotiff_url,
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.6
        )
        raster.add_to(m)
        folium.LayerControl().add_to(m)
        return m
    except Exception as e:
        st.error(f"Failed to display GeoTIFF on the map: {e}")
        return None

# Main processing and mapping
rain_data = fetch_rain_data(start_time, end_time)

geotiff_path = fetch_rain_data_as_geotiff(rain_data)
if geotiff_path:
    st.write("GeoTIFF created at:", geotiff_path)
    
    # Serve the GeoTIFF and get the URL
    geotiff_url, httpd = serve_geotiff(geotiff_path)
    
    # Display the map
    m = map_geotiff(geotiff_url)
    if m:
        st.components.v1.html(m._repr_html_(), height=500)
    
    # Allow the user to download the GeoTIFF file
    with open(geotiff_path, "rb") as file:
        st.download_button(
            label="Download GeoTIFF",
            data=file,
            file_name="rainrate_geotiff.tif",
            mime="image/tiff"
        )
    
    # Stop the HTTP server when done
    httpd.shutdown()
else:
    st.error("Failed to create GeoTIFF.")
