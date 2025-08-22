# Auto-generated from: NYCSBUS_Trial_EDA.ipynb
# Run with:   streamlit run nycsbus_streamlit_app.py

import streamlit as st
st.set_page_config(page_title="NYCSBUS EDA", layout="wide")
st.title("NYCSBUS: Trial EDA (auto-converted)")

# --- Helpers to make notebook-style output show in Streamlit ---
import builtins as __builtins__

def _st_display(obj):
    """
    Best-effort display hook: tries Streamlit-first, falls back to print.
    This makes notebook calls like `display(df)` render in the app.
    """
    try:
        import pandas as pd
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            st.dataframe(obj)
            return
    except Exception:
        pass
    try:
        st.write(obj)
    except Exception:
        try:
            print(obj)
        except Exception:
            pass

__builtins__.display = _st_display  # so notebook display() calls still work

def _show_streamlit_figures():
    """
    After any plotting code, show all active matplotlib figures in Streamlit,
    then close them so we don't duplicate on reruns.
    """
    try:
        import matplotlib.pyplot as plt
        figs = [plt.figure(i) for i in plt.get_fignums()]
        for fig in figs:
            st.pyplot(fig)
        plt.close('all')
    except Exception:
        pass

st.markdown("> **Note:** This app was auto-converted from a notebook. "
            "If some inline outputs (like bare `df`) do not appear, wrap them in `display(...)` or `st.write(...)`.")

# %% [code] - cell 0
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    !pip install h3 pandas geopandas esda libpysal pymannkendall
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 1
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    !pip install h3==3.7.6
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Step 1: Data Loading and Cleaning**
""")

# %% [markdown]
st.markdown("""
*   NYC crash data is loaded
*   Dates are converted to datetime format
*   Rows missing location (LATITUDE, LONGITUDE) or time (CRASH DATE) are removed, since we need both to assign H3 bins.
""")

# %% [code] - cell 4
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    import pandas as pd
    import numpy as np
    import h3
    import geopandas as gpd
    from shapely.geometry import Polygon
    from libpysal.weights import KNN
    from esda.getisord import G_Local
    import pymannkendall as mk
    import matplotlib.pyplot as plt
    import seaborn as sns
    import folium
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 5
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 6
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    df = pd.read_csv('/content/drive/My Drive/NYCSBUS Project/Motor_Vehicle_Collisions_-_Crashes_20250715.csv', low_memory=False)
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 7
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    df.head()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 8
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    print(df.columns)
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 9
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'], errors='coerce')
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'CRASH DATE'])
    print("Data loaded and cleaned:")
    df.head()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Step 2: Spatial and Temporal Binning**
""")

# %% [markdown]
st.markdown("""
*   Each event is mapped to an H3 hexagon (resolution=9, suitable for city-scale).
*   Dates are grouped into daily intervals.
""")

# %% [code] - cell 12
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    resolution = 9
    df['h3_cell'] = df.apply(lambda row: h3.geo_to_h3(row['LATITUDE'], row['LONGITUDE'], resolution), axis=1)
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 13
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    df['time_bin'] = df['CRASH DATE'].dt.floor('D')
    print("Data with H3 cells and time bins:")
    df[['LATITUDE', 'LONGITUDE', 'h3_cell', 'time_bin']].head()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Step 3: Aggregating Events Per H3 Cell Per Day**
""")

# %% [markdown]
st.markdown("""
*   We count how many crashes happened per (H3 cell, day).
*   The H3 hexagon IDs are converted to polygon shapes for visualization and spatial analysis.
""")

# %% [code] - cell 16
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    grouped = df.groupby(['h3_cell', 'time_bin']).size().reset_index(name='event_count')
    display(grouped.head())
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 17
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    def h3_to_polygon(h3_cell):
        boundary = h3.h3_to_geo_boundary(h3_cell, geo_json=True)
        return Polygon(boundary)
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 18
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    grouped['geometry'] = grouped['h3_cell'].apply(h3_to_polygon)
    gdf = gpd.GeoDataFrame(grouped, geometry='geometry')
    print("Event counts with geometry:")
    gdf.head()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Step 4: Trend Detection using Mann-Kendall Test**
""")

# %% [markdown]
st.markdown("""
*   For each H3 cell’s time series, we apply the Mann-Kendall test to identify trends: Increasing, Decreasing, or No Trend.
*   Ignore short time series (< 3 points)
""")

# %% [code] - cell 21
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    trend_results = []

    for h3_cell, sub_df in grouped.groupby('h3_cell'):
        counts = sub_df.sort_values('time_bin')['event_count']
        if len(counts) >= 3:
            trend_test = mk.original_test(counts)
            trend_results.append({'h3_cell': h3_cell, 'trend': trend_test.trend, 'p': trend_test.p})

    trend_df = pd.DataFrame(trend_results)
    print("Trend detection results:")
    trend_df.head()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Gi Hotspot Detection & Alarm Classification (Daily)**
""")

# %% [markdown]
st.markdown("""
*   For each day:

1.   Build a spatial neighborhood (KNN) for Gi* analysis.

2.   Compute Gi z-scores* to detect spatial hotspots.

1.   Classify alarms based on:z-score significance (>1.96 = 95% confidence), time trend, combination rules (like new/persistent hotspots).
""")

# %% [code] - cell 24
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    from tqdm import tqdm
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 25
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    print("Starting Gi* Hotspot Detection and Alarm Classification...")
    alarm_records = []

    dates = gdf['time_bin'].unique()

    for date in tqdm(dates, desc="Processing days"):
        day_data = gdf[gdf['time_bin'] == date].copy()
        coords = np.array([(geom.centroid.y, geom.centroid.x) for geom in day_data['geometry']])

        if len(coords) < 6:
            continue

        w = KNN.from_array(coords, k=6)
        gi_star = G_Local(day_data['event_count'].values.astype(float), w)
        day_data['gi_z'] = gi_star.Zs

        # Vectorized merge and alarm classification
        day_data = day_data.merge(trend_df, on='h3_cell', how='left')

        conditions = [
            (day_data['gi_z'] > 1.96) & (day_data['trend'] == 'increasing'),
            (day_data['gi_z'] > 1.96) & (day_data['trend'] == 'no trend'),
            (day_data['gi_z'] > 1.96) & (day_data['trend'] == 'decreasing'),
            (day_data['gi_z'] > 1.96)
        ]

        choices = [
            'Intensifying Hotspot',
            'Persistent Hotspot',
            'Diminishing Hotspot',
            'New Hotspot'
        ]

        day_data['alarm_type'] = np.select(conditions, choices, default=None)

        day_alarms = day_data.dropna(subset=['alarm_type'])
        day_alarms['time_bin'] = date
        alarm_records.append(day_alarms)

    final_alarm_table = pd.concat(alarm_records, ignore_index=True)
    print("\nFinal Combined Alarm Table:")
    final_alarm_table[['h3_cell', 'time_bin', 'alarm_type', 'event_count', 'trend', 'gi_z']]
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Summary Visualizations**
""")

# %% [code] - cell 27
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    sns.countplot(x='trend', data=trend_df)
    plt.title('Trend Classification Across H3 Cells')
    plt.show()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 28
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    sns.countplot(x='alarm_type', data=final_alarm_table)
    plt.title('Alarm Types Frequency')
    plt.show()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 29
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    sns.histplot(final_alarm_table['gi_z'], bins=30, kde=True)
    plt.title('Gi* Z-score Distribution Across Alarms')
    plt.show()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Folium Hotspot Map**
""")

# %% [code] - cell 31
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    import folium
    from folium import plugins
    from IPython.display import display
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 32
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    map_center = [40.75, -73.98]
    hotspot_map = folium.Map(location=map_center, zoom_start=11)

    for _, row in final_alarm_table.iterrows():
        color = 'red' if 'Hotspot' in row['alarm_type'] else 'blue'
        geo_json = {'type': 'Polygon', 'coordinates': [list(row['geometry'].exterior.coords)]}
        folium.GeoJson(
            geo_json,
            style_function=lambda x, color=color: {'color': color, 'weight': 1, 'fillOpacity': 0.4}
        ).add_to(hotspot_map)
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 33
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    display(hotspot_map)
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 34
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    def classify_intensity(z_score):
        if z_score > 2.5:
            return 'High'
        elif z_score > 1.96:
            return 'Medium'
        else:
            return 'Low'

    final_alarm_table['intensity_level'] = final_alarm_table['gi_z'].apply(classify_intensity)

    intensity_summary = final_alarm_table.groupby('intensity_level').agg(
        num_areas=('h3_cell', 'nunique'),
        total_events=('event_count', 'sum')
    ).reset_index()

    print("Hotspot Intensity Summary Table:")
    intensity_summary
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 35
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    map_center = [40.75, -73.98]  # NYC center
    hotspot_map = folium.Map(location=map_center, zoom_start=11)

    for _, row in final_alarm_table.iterrows():
        color = 'red' if 'Hotspot' in row['alarm_type'] else 'blue'
        geo_json = {'type': 'Polygon', 'coordinates': [list(row['geometry'].exterior.coords)]}
        folium.GeoJson(geo_json, style_function=lambda x, color=color: {'color': color, 'weight': 1}).add_to(hotspot_map)

    hotspot_map.save('all_hotspots_map.html')
    print("Hotspot map saved as 'all_hotspots_map.html'")
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [markdown]
st.markdown("""
# **Example Trend Plot for One H3 Cell**
""")

# %% [code] - cell 37
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    example_cell = final_alarm_table['h3_cell'].iloc[0]
    cell_data = grouped[grouped['h3_cell'] == example_cell].sort_values('time_bin')

    plt.figure(figsize=(10,5))
    plt.grid(True)
    plt.plot(cell_data['time_bin'], cell_data['event_count'], marker='o')
    plt.title(f'Event Count Trend for H3 Cell: {example_cell}')
    plt.xlabel('Date')
    plt.ylabel('Event Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

# %% [code] - cell 38
# You can add st.sidebar controls above to parameterize this cell's logic.
try:
    final_alarm_table
except Exception as __e:
    st.exception(__e)

# Show any matplotlib figures produced by this cell
_show_streamlit_figures()

st.markdown("---")
st.subheader("Tips")
st.markdown(
    "- Use the **left sidebar** to add inputs and controls (e.g., file upload, sliders).\n"
    "- Replace print statements with `st.write` or `display` for rich rendering.\n"
    "- If you create plots with matplotlib, they should appear automatically after each cell."
)
