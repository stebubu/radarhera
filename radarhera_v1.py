import streamlit as st
import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import tempfile
import requests
from datetime import datetime, timedelta
import os
from mapboxgl.viz import RasterTilesViz
from mapboxgl.utils import create_color_stops
from mapboxgl.viz import RasterTilesViz
import json
import leafmap.foliumap as leafmap

from matplotlib import cm, colors
from branca.colormap import linear
import pytz

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



# Display the logo in the header
st.image("logo.png", width=200)  # Adjust width as needed

# Title and Sidebar
st.title("HERA RADAR")
st.sidebar.title("Settings")

# Date selection
selected_date = st.sidebar.date_input("Select date", datetime.utcnow().date())

# Hour selection
selected_hour = st.sidebar.selectbox("Select hour of the day in UTC Time ZONE", options=range(24), index=datetime.utcnow().hour)

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
# Convert UTC time to CET time
cet = pytz.timezone('CET')
end_time_cet = end_time.astimezone(cet)
start_time_cet = start_time.astimezone(cet)
# Fetch data from API and accumulate rain data
def fetch_acc_rain_data(start_time, end_time):
    current_time = start_time
    st.write(f"'start UTC Time {start_time}")
    st.write(f"'end UTC TIME {end_time}")
    st.write(f"'start CET Time {start_time_cet}")
    st.write(f"'end CET TIME {end_time_cet}")
    
    accumulated_rain = None
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
                
                # Assuming the rain data is in a variable named 'rainrate'
                if 'rainrate' in ds.variables:
                    rain = ds['rainrate'].squeeze()  # Remove any singleton dimensions

                    # Align the datasets before summing
                    if accumulated_rain is None:
                        accumulated_rain = rain
                        #st.write(f"Current time: {current_time}")
                        #st.write(f"Rainrate shape: {rain.shape}")
                        #st.write(f"Rainrate sum: {rain.sum()}")
                        #st.write(f"Accumulated rain dimensions: {accumulated_rain.dims}")
                        #st.write(f"Accumulated rain shape before sum: {accumulated_rain.shape}")

                    else:
                        # Align the datasets using 'outer' join
                        accumulated_rain, rain = xr.align(accumulated_rain, rain, join='outer')
                        accumulated_rain = accumulated_rain + rain.fillna(0)
                        #st.write(f"Current time: {current_time}")
                        #st.write(f"Rainrate shape: {rain.shape}")
                        #st.write(f"Rainrate sum: {rain.sum()}")
                        #st.write(f"Accumulated rain dimensions: {accumulated_rain.dims}")
                        #st.write(f"Accumulated rain shape before sum: {accumulated_rain.shape}")

                else:
                    st.error(f"'rainrate' variable not found in dataset for {current_time}")
                
            except Exception as e:
                st.error(f"Failed to open dataset: {e}")
        else:
            st.error(f"Error fetching data for {current_time}: {response.text}")
            break
        
        current_time += timedelta(minutes=5)
    
    # Clean up temporary files
    for file_path in temp_files:
        #st.error(f"Cleaning: {file_path}")
        try:
            os.remove(file_path)
        except Exception as e:
            st.error(f"Failed to remove temporary file: {file_path}. Error: {e}")
    # Final processing to sum across the first dimension (7 slices)
    if accumulated_rain is not None:
        #accumulated_rain = accumulated_rain.sum(dim='time')  # Replace 'dim_0' with the actual dimension name if available
        #st.write(f"Accumulated rain dimensions before squeeze: {accumulated_rain.dims}")
        #st.write(f"Accumulated rain shape before squeeze: {accumulated_rain.shape}")
        # Ensure that the result is a 2D array (lat, lon)
        #accumulated_rain = accumulated_rain.squeeze()
        #st.write(f"somma finale: {accumulated_rain.sum()}")
        
        # Perform vertical flip
        accumulated_rain = accumulated_rain[::-1, :]  # Flip vertically along the first dimension (latitude)


    return accumulated_rain#, tmp_file_path            
     
    



def convert_accumulated_rain_to_geotiff(accumulated_rain):
    if accumulated_rain is not None and accumulated_rain.size > 0:
        # Handle potential extra dimensions in rainrate
        rainrate = accumulated_rain.squeeze().values  # Remove any singleton dimensions

        # If rainrate still has more than 2 dimensions, select the first slice
        if rainrate.ndim > 2:
            st.warning(f"Rainrate has {rainrate.shape[0]} slices, selecting the first one.")
            rainrate = rainrate[0, :, :]

        # Extract lat, lon from the accumulated rain DataArray
        lat = accumulated_rain.coords['lat'].values
        lon = accumulated_rain.coords['lon'].values

        # Print shapes for debugging
        #st.write(f"Latitude shape: {lat.shape}")
        #st.write(f"Longitude shape: {lon.shape}")
        #st.write(f"Rainrate shape: {rainrate.shape}")

        # Use np.meshgrid to align lat and lon with rainrate
        lon, lat = np.meshgrid(lon, lat)

        # Re-check shapes after meshgrid/alignment
        #st.write(f"After meshgrid/alignment - Latitude shape: {lat.shape}")
        #st.write(f"After meshgrid/alignment - Longitude shape: {lon.shape}")

        if lat.shape == lon.shape == rainrate.shape:
            # Calculate geographic transform parameters
            lon_min, lat_max = lon.min(), lat.max()
            
            cell_size_lon = (lon.max() - lon.min()) / lon.shape[1]
            cell_size_lat = (lat.max() - lat.min()) / lat.shape[0]
            #st.write(f"cell_size_lon: {str(cell_size_lon)}")
            #st.write(f"lon_min: {str(lon_min)}")
            #st.write(f"lat_max: {str(lat_max)}")
            #st.write(f"cell_size_lat: {str(cell_size_lat)}")

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
                    dtype=rasterio.float32,  # Correctly use the rasterio data type
                    crs='EPSG:4326',
                    transform=transform,
                ) as dst:
                    dst.write(rainrate.astype(rasterio.float32), 1)
                geotiff_path = tmp_file.name
                return geotiff_path
        else:
            st.error("Mismatch in array dimensions: lat, lon, and rainrate must have the same shape.")
            return None
    else:
        st.warning("No data available for the selected time and cumulative interval.")
        return None

# Convert GeoTIFF to Cloud Optimized GeoTIFF (COG)
def convert_to_cog(geotiff_path):
    cog_path = geotiff_path.replace(".tif", "_cog.tif")
    with rasterio.open(geotiff_path) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="deflate",
            interleave="band",
            dtype=rasterio.float32,
            bigtiff="IF_SAFER",
        )
        with rasterio.open(cog_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                data = src.read(i, resampling=Resampling.nearest)
                dst.write(data, indexes=i)
    return cog_path

# Display COG using Mapbox GL

# Function to display the raster on a Mapbox basemap using leafmap
import streamlit as st
from mapboxgl.viz import RasterTilesViz
import os




def inspect_cog(cog_path):
    try:
        st.write("Inspecting COG...")

        with rasterio.open(cog_path) as src:
            bounds = src.bounds
            crs = src.crs
            count = src.count

            #st.write(f"Bounds: {bounds}")
            #st.write(f"CRS: {crs}")
            #st.write(f"Number of bands: {count}")
            
            # Read the first band just to ensure we can read the data
            band1 = src.read(1)
            #st.write(f"Shape of Band 1: {band1.shape}")
            #st.write(f"Data Type of Band 1: {band1.dtype}")

    except Exception as e:
        st.error(f"Failed to inspect COG: {e}")
        st.write(f"Error details: {str(e)}")
import streamlit as st
import rasterio
import folium
from rasterio.plot import reshape_as_image
from rasterio.warp import transform_bounds
from streamlit_folium import folium_static

def display_cog_with_folium(cog_path):
    try:
        st.write("Rendering map with Folium...")

        with rasterio.open(cog_path) as src:
            # Ensure the COG is in EPSG:4326 (WGS 84)
            bounds = src.bounds
            crs = src.crs
            count = src.count

            # Transform bounds to lat/lon
            lon_min, lat_min, lon_max, lat_max = bounds
            #st.write(f"Bounds (Lat/Lon): {(lat_min, lon_min), (lat_max, lon_max)}")

            # Read the first band (assuming single-band raster for simplicity)
            band1 = src.read(1)

            # Check if the array is empty or all zeros
            if band1.size == 0 or np.all(band1 == 0):
                st.error("The raster data is empty or contains only zero values.")
                #return

            # Reshape as image if needed (for multi-band)
            if count > 1:
                image = reshape_as_image(src.read([1, 2, 3]))  # Assuming RGB bands
            else:
                image = band1  # For single band, use directly

            # Create a normalized colormap where 0 is transparent and blue is the color
            vmin = np.min(band1[band1 > 0]) if np.any(band1 > 0) else 0
            vmax = np.max(band1)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap('Blues')
            # Create a normalized colormap where 0 is transparent and blue is the color
            #norm = colors.Normalize(vmin=np.min(band1[band1 > 0]), vmax=np.max(band1))
            #cmap = cm.get_cmap('Blues')

            # Apply the colormap and transparency
            rgba_image = cmap(norm(band1))
            rgba_image[band1 == 0] = [0, 0, 0, 0]  # Make value 0 fully transparent


            # Create a folium map centered on the raster
            m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=10)

            # Add the raster as an image overlay
            image_overlay = folium.raster_layers.ImageOverlay(
                image=rgba_image,
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                opacity=0.7,
                interactive=True,
                cross_origin=False,
                zindex=1
            )
            image_overlay.add_to(m)
            
            # Add a color legend
            colormap = linear.Blues_09.scale(vmin, vmax)
            colormap.caption = 'Rainfall Intensity'
            colormap.add_to(m)


        

            # Add a popup to show lat, lon, and value on click
            #m.add_child(folium.LatLngPopup())
                      
            #folium.ClickForMarker(popup="Click to see lat/lon").add_to(m)


            # Render the map in Streamlit
            folium_static(m)

    except Exception as e:
        st.error(f"Failed to display COG with Folium: {e}")
        st.write(f"Error details: {str(e)}")


# Main processing and mapping
#rain_data = fetch_rain_data(start_time, end_time)
rain_data = fetch_acc_rain_data(start_time, end_time)
#geotiff_path = fetch_rain_data_as_geotiff(rain_data)

geotiff_path =convert_accumulated_rain_to_geotiff(rain_data)


if geotiff_path:
    st.write("GeoTIFF created at:", geotiff_path)
    
    # Convert GeoTIFF to COG
    cog_path = convert_to_cog(geotiff_path)
    st.write("COG created at:", cog_path)
    inspect_cog(cog_path)
    
    display_cog_with_folium(cog_path)
    #display_cog_on_map(cog_path)
            # Allow the user to download the COG file
    with open(cog_path, "rb") as file:
        st.download_button(
            label="Download COG",
            data=file,
            file_name="rainrate_cog.tif",
            mime="image/tiff"
        )
    
else:
    st.error("Failed to create GeoTIFF.")


