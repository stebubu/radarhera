import streamlit as st
import requests
import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pydeck as pdk
import pandas as pd
import tempfile

# Set up the API authentication
auth_url = "https://api.hypermeteo.com/auth-b2b/authenticate"
body = {
    "username": "gecosistema",
    "password": "Saferplaces2023!"
}
response = requests.post(auth_url, json=body).json()
token = response['token']

# Set up headers with token
headers = {"Authorization": f"Bearer {token}"}

# WCS Request parameters
base_url = "https://api.hypermeteo.com/b2b-binary/ogc/geoserver/wcs"
service = "WCS"
request_type = "GetCoverage"
version = "2.0.0"
coverage_id = "RADAR_HERA_150M_5MIN__rainrate"
format_type = "application/x-netcdf"
subset_lat = "Lat(43.8,44.2)"
subset_lon = "Long(12.4,12.9)"

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
            # Save the response content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name

            try:
                # Open the dataset from the temporary file
                ds = xr.open_dataset(tmp_file_path, engine='netcdf4')  # You can change the engine if needed
                rain_data.append(ds)
            except Exception as e:
                st.error(f"Failed to open dataset: {e}")
            finally:
                os.remove(tmp_file_path)  # Clean up the temporary file

        else:
            st.error(f"Error fetching data for {current_time}: {response.text}")
            return None
        
        current_time += timedelta(minutes=5)
    
    return rain_data

rain_data = fetch_rain_data(start_time, end_time)

# Check if rain_data is not None and not empty
if rain_data and len(rain_data) > 0:
    if cumulative_interval != "No Cumulative":
        # Combine data if necessary, for cumulative sum
        combined_data = xr.concat(rain_data, dim='time').sum(dim='time')
    else:
        combined_data = rain_data[0]  # Use the single time step data

    # Extract and align lat, lon, and rainrate
    try:
        lat = combined_data.coords['lat'].values
        lon = combined_data.coords['lon'].values
        rainrate = combined_data['rainrate'].values

        # Ensure that lat, lon, and rainrate are 2D and have the same shape
        if len(lat.shape) == 1 and len(lon.shape) == 1 and len(rainrate.shape) == 2:
            lon, lat = np.meshgrid(lon, lat)
        
        if lat.shape == lon.shape == rainrate.shape:
            # Flatten the arrays for plotting
            df = pd.DataFrame({
                'lat': lat.flatten(),
                'lon': lon.flatten(),
                'rainrate': rainrate.flatten()
            })

            # Ensure the latitude and longitude are valid numbers
            mean_lat = np.mean(lat)
            mean_lon = np.mean(lon)

            if np.isfinite(mean_lat) and np.isfinite(mean_lon):
                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=float(mean_lat),
                        longitude=float(mean_lon),
                        zoom=8,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            'ScatterplotLayer',
                            data=df,
                            get_position='[lon, lat]',
                            get_color='[200, 30, 0, 160]',
                            get_radius='rainrate * 100',
                            pickable=True,
                        ),
                    ],
                ))
            else:
                st.error("Invalid latitude or longitude values.")
        else:
            st.error("Mismatch in array dimensions: lat, lon, and rainrate must have the same shape.")
    except KeyError as e:
        st.error(f"Missing expected data in the dataset: {e}")
else:
    st.warning("No data available for the selected time and cumulative interval.")

