
# nycsbus_streamlit_app.py
# -------------------------------------------------------------
# NYC Spatiotemporal Crash Analysis (Streamlit app)
# Lightweight, visually appealing app that avoids Jupyter magics
# and heavy native deps. Works with CSVs you already have:
#   - hotspot_analysis.csv
#   - trend_analysis.csv
#   - spatiotemporal_alarms.csv
#
# How to run locally:
#   1) pip install streamlit pandas pydeck altair
#   2) streamlit run nycsbus_streamlit_app.py
#
# On Streamlit Cloud:
#   - Add these CSVs to your repo root (same folder as this .py).
#   - (Optional) Create a requirements.txt with:
#       streamlit
#       pandas
#       pydeck
#       altair
# -------------------------------------------------------------

import os
import sys
import math
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt
import streamlit as st

st.set_page_config(
    page_title='NYC Spatiotemporal Crash Analysis',
    page_icon='🗽',
    layout='wide'
)

# --------------------------
# Small style polish
# --------------------------
st.markdown(
    '''
    <style>
      /* Reduce top padding */
      .block-container { padding-top: 1rem; }

      /* Metric cards */
      div[data-testid="stMetricValue"] { font-size: 1.8rem; }
      .metric-card {
        padding: 0.9rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(49,51,63,0.2);
        background: rgba(250,250,250,0.65);
      }
      /* Section headers */
      .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.2rem 0 0.6rem 0;
      }
    </style>
    ''',
    unsafe_allow_html=True
)

# --------------------------
# Helpers
# --------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        # try with latin-1 fallback
        return pd.read_csv(path, encoding='latin-1')


def add_color_columns_for_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    '''Create RGB columns for pydeck color mapping based on hotspot_type / gi_star.'''
    if df is None or df.empty:
        return df
    out = df.copy()
    if 'hotspot_type' in out.columns:
        # Map: Hot Spot = red-ish, Normal = gray
        out['fill_r'] = np.where(out['hotspot_type'].str.contains('Hot', case=False, na=False), 220, 160)
        out['fill_g'] = np.where(out['hotspot_type'].str.contains('Hot', case=False, na=False), 60, 160)
        out['fill_b'] = np.where(out['hotspot_type'].str.contains('Hot', case=False, na=False), 60, 160)
    else:
        # Fallback by gi_star sign
        out['fill_r'] = np.where(out.get('gi_star', 0) > 0, 220, 160)
        out['fill_g'] = np.where(out.get('gi_star', 0) > 0, 60, 160)
        out['fill_b'] = np.where(out.get('gi_star', 0) > 0, 60, 160)
    return out


def add_color_columns_for_trends(df: pd.DataFrame) -> pd.DataFrame:
    '''Create RGB columns for pydeck color mapping based on trend direction/significance.'''
    if df is None or df.empty:
        return df
    out = df.copy()
    direction = out.get('trend_direction', pd.Series([''] * len(out)))
    significance = out.get('trend_significance', pd.Series([''] * len(out)))
    # Increasing & significant: green; Decreasing & significant: orange; otherwise gray
    inc = direction.str.contains('increase', case=False, na=False)
    dec = direction.str.contains('decrease', case=False, na=False)
    sig = significance.str.contains('significant', case=False, na=False) & ~significance.str.contains('not', case=False, na=False)
    out['fill_r'] = np.where(sig & dec, 230, np.where(sig & inc, 80, 150))
    out['fill_g'] = np.where(sig & dec, 140, np.where(sig & inc, 200, 150))
    out['fill_b'] = np.where(sig & dec, 60,  np.where(sig & inc, 80, 150))
    return out


def center_on(df: pd.DataFrame, lat_col='center_lat', lon_col='center_lon', default=(40.7128, -74.0060)):
    '''Compute a reasonable center for the map.'''
    if df is None or df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return default
    lat = df[lat_col].astype(float).clip(-90, 90)
    lon = df[lon_col].astype(float).clip(-180, 180)
    return float(lat.mean()), float(lon.mean())


def as_time_order(series: pd.Series) -> pd.Series:
    '''Attempt to order time bins sensibly; if not parseable, return as is.'''
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        if parsed.notna().any():
            # Replace with parsed when possible
            order = pd.Series(series, index=series.index)
            order.loc[parsed.notna()] = parsed.dt.strftime('%Y-%m-%d')
            return order
        return series
    except Exception:
        return series


# --------------------------
# Data loading
# --------------------------
hotspots = load_csv('hotspot_analysis.csv')
trends = load_csv('trend_analysis.csv')
alarms = load_csv('spatiotemporal_alarms.csv')

# If running in a new repo, also allow uploads via sidebar
with st.sidebar:
    st.header('📂 Data')
    st.caption('Drop CSVs here if they are not already in the same folder as this app.')
    up_hot = st.file_uploader('hotspot_analysis.csv', type=['csv'], accept_multiple_files=False)
    up_trd = st.file_uploader('trend_analysis.csv', type=['csv'], accept_multiple_files=False)
    up_alm = st.file_uploader('spatiotemporal_alarms.csv', type=['csv'], accept_multiple_files=False)

if up_hot is not None:
    hotspots = pd.read_csv(up_hot)
if up_trd is not None:
    trends = pd.read_csv(up_trd)
if up_alm is not None:
    alarms = pd.read_csv(up_alm)

# Normalize column names
def clean_cols(df):
    return df.rename(columns={c: c.strip().replace(' ', '_') for c in df.columns})

if hotspots is not None:
    hotspots = clean_cols(hotspots)
if trends is not None:
    trends = clean_cols(trends)
if alarms is not None:
    alarms = clean_cols(alarms)

# Validate expected columns and gently adapt
# - Hotspots: h3_cell, time_bin, crash_count, center_lat, center_lon, gi_star, gi_p_value, hotspot_type
# - Trends:   h3_cell, center_lat, center_lon, trend_slope, trend_p_value, trend_significance, trend_direction, total_crashes
# - Alarms:   H3_Cell_ID, Centroid_Lat, Centroid_Lon, Time_Period, Alarm_Type, Crash_Count, Gi_Star_Score, Priority, Recommended_Action, Hot_Period_Count
if alarms is not None:
    # Standardize naming to match other refs
    rename_map = {
        'H3_Cell_ID': 'h3_cell',
        'Centroid_Lat': 'center_lat',
        'Centroid_Lon': 'center_lon',
        'Time_Period': 'time_bin',
        'Alarm_Type': 'alarm_type',
        'Crash_Count': 'crash_count',
        'Gi_Star_Score': 'gi_star',
        'Priority': 'priority',
        'Recommended_Action': 'recommended_action',
        'Hot_Period_Count': 'hot_period_count',
    }
    alarms = alarms.rename(columns=rename_map)

# Add color columns
hotspots_c = add_color_columns_for_hotspots(hotspots) if hotspots is not None else None
trends_c = add_color_columns_for_trends(trends) if trends is not None else None

# --------------------------
# Sidebar filters
# --------------------------
with st.sidebar:
    st.header('🎛️ Filters')
    # Time filter (works for both hotspots and alarms, if available)
    time_options = []
    if hotspots_c is not None and 'time_bin' in hotspots_c.columns:
        time_options.extend(list(hotspots_c['time_bin'].dropna().unique()))
    if alarms is not None and 'time_bin' in alarms.columns:
        time_options.extend(list(alarms['time_bin'].dropna().unique()))
    time_options = sorted(pd.unique(pd.Series(time_options)), key=lambda x: str(x))

    selected_times = st.multiselect('Time Period(s)', options=time_options, default=time_options[:1] if time_options else [])

    hotspot_types = []
    if hotspots_c is not None and 'hotspot_type' in hotspots_c.columns:
        hotspot_types = sorted(hotspots_c['hotspot_type'].dropna().unique().tolist())

    selected_hotspot_types = st.multiselect('Hotspot Type', hotspot_types, default=hotspot_types)

    trend_dirs = []
    if trends_c is not None and 'trend_direction' in trends_c.columns:
        trend_dirs = sorted(trends_c['trend_direction'].dropna().unique().tolist())

    selected_trend_dirs = st.multiselect('Trend Direction', trend_dirs, default=trend_dirs)

    trend_signifs = []
    if trends_c is not None and 'trend_significance' in trends_c.columns:
        trend_signifs = sorted(trends_c['trend_significance'].dropna().unique().tolist())

    selected_trend_signifs = st.multiselect('Trend Significance', trend_signifs, default=trend_signifs)

    alarm_types = []
    if alarms is not None and 'alarm_type' in alarms.columns:
        alarm_types = sorted(alarms['alarm_type'].dropna().unique().tolist())

    selected_alarm_types = st.multiselect('Alarm Type', alarm_types, default=alarm_types)

    max_rows = st.slider('Max rows to show in tables', 50, 5000, 500, 50)

# Apply filters
if hotspots_c is not None:
    hs = hotspots_c.copy()
    if selected_times:
        if 'time_bin' in hs.columns:
            hs = hs[hs['time_bin'].isin(selected_times)]
    if selected_hotspot_types:
        if 'hotspot_type' in hs.columns:
            hs = hs[hs['hotspot_type'].isin(selected_hotspot_types)]
else:
    hs = None

if trends_c is not None:
    tr = trends_c.copy()
    if selected_trend_dirs:
        if 'trend_direction' in tr.columns:
            tr = tr[tr['trend_direction'].isin(selected_trend_dirs)]
    if selected_trend_signifs:
        if 'trend_significance' in tr.columns:
            tr = tr[tr['trend_significance'].isin(selected_trend_signifs)]
else:
    tr = None

if alarms is not None:
    al = alarms.copy()
    if selected_times:
        if 'time_bin' in al.columns:
            al = al[al['time_bin'].isin(selected_times)]
    if selected_alarm_types:
        if 'alarm_type' in al.columns:
            al = al[al['alarm_type'].isin(selected_alarm_types)]
else:
    al = None

# --------------------------
# Header
# --------------------------
st.title('🗽 NYC Spatiotemporal Crash Analysis')
st.caption('Interactive overview of hotspots (Gi*), trends, and alarms synthesized from H3-binned crash data.')

# --------------------------
# KPI Row
# --------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    with st.container(border=True):
        st.markdown('<div class="section-title">Total Crashes (Trend Data)</div>', unsafe_allow_html=True)
        total_crashes = int(tr['total_crashes'].sum()) if (tr is not None and 'total_crashes' in tr.columns) else 0
        st.metric(label='', value=f"{total_crashes:,}")

with col2:
    with st.container(border=True):
        st.markdown('<div class="section-title">Hot Spots</div>', unsafe_allow_html=True)
        hot_count = int((hs['hotspot_type'].str.contains('Hot', case=False)).sum()) if (hs is not None and 'hotspot_type' in hs.columns) else 0
        st.metric(label='', value=f"{hot_count:,}")

with col3:
    with st.container(border=True):
        st.markdown('<div class="section-title">Significant ↑ Trends</div>', unsafe_allow_html=True)
        inc_sig = 0
        if tr is not None and 'trend_direction' in tr.columns and 'trend_significance' in tr.columns:
            inc = tr['trend_direction'].str.contains('increase', case=False, na=False)
            sig = tr['trend_significance'].str.contains('significant', case=False, na=False) & ~tr['trend_significance'].str.contains('not', case=False, na=False)
            inc_sig = int((inc & sig).sum())
        st.metric(label='', value=f"{inc_sig:,}")

with col4:
    with st.container(border=True):
        st.markdown('<div class="section-title">Active Alarms</div>', unsafe_allow_html=True)
        alarm_count = int(len(al)) if al is not None else 0
        st.metric(label='', value=f"{alarm_count:,}")

st.divider()

# --------------------------
# Tabs
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(['🗺️ Hotspots Map', '📈 Trends', '🚨 Alarms', '📊 Overview', '🧾 Data'])

# --- Tab 1: Hotspots Map ---
with tab1:
    st.subheader('Hotspots (H3)')
    if hs is None or hs.empty:
        st.info('No hotspot data loaded.')
    else:
        # Prefer H3HexagonLayer (no Python h3 needed). If unavailable on the front-end, we fall back to center points.
        map_center = center_on(hs, 'center_lat', 'center_lon', default=(40.72, -74.0))
        initial = pdk.ViewState(latitude=map_center[0], longitude=map_center[1], zoom=10, pitch=40)

        try:
            h3_layer = pdk.Layer(
                'H3HexagonLayer',
                data=hs,
                get_hexagon='h3_cell',
                get_fill_color='[fill_r, fill_g, fill_b]',
                get_elevation='crash_count',
                elevation_scale=50,
                extruded=True,
                pickable=True,
                opacity=0.8,
            )
            deck = pdk.Deck(layers=[h3_layer], initial_view_state=initial,
                            tooltip={'text': 'Cell: {h3_cell}\nCrashes: {crash_count}\nGi*: {gi_star}\nType: {hotspot_type}'})
            st.pydeck_chart(deck, use_container_width=True)
        except Exception:
            # Fallback: center points as columns
            scatter = pdk.Layer(
                'ScatterplotLayer',
                data=hs,
                get_position='[center_lon, center_lat]',
                get_radius='100 + crash_count * 40',
                get_fill_color='[fill_r, fill_g, fill_b]',
                pickable=True,
                opacity=0.7,
            )
            deck = pdk.Deck(layers=[scatter], initial_view_state=initial,
                            tooltip={'text': 'Cell: {h3_cell}\nCrashes: {crash_count}\nGi*: {gi_star}\nType: {hotspot_type}'})
            st.pydeck_chart(deck, use_container_width=True)

# --- Tab 2: Trends ---
with tab2:
    st.subheader('Trend Analysis')
    if tr is None or tr.empty:
        st.info('No trend data loaded.')
    else:
        c1, c2 = st.columns([3, 2])

        with c1:
            # Trend map
            map_center = center_on(tr, 'center_lat', 'center_lon', default=(40.72, -74.0))
            initial = pdk.ViewState(latitude=map_center[0], longitude=map_center[1], zoom=10, pitch=30)

            try:
                h3_layer = pdk.Layer(
                    'H3HexagonLayer',
                    data=tr,
                    get_hexagon='h3_cell',
                    get_fill_color='[fill_r, fill_g, fill_b]',
                    get_elevation='total_crashes',
                    elevation_scale=30,
                    extruded=True,
                    pickable=True,
                    opacity=0.8,
                )
                deck = pdk.Deck(layers=[h3_layer], initial_view_state=initial,
                                tooltip={'text': 'Cell: {h3_cell}\nTrend: {trend_direction}\nSignificance: {trend_significance}\nTotal crashes: {total_crashes}'})
                st.pydeck_chart(deck, use_container_width=True)
            except Exception:
                scatter = pdk.Layer(
                    'ScatterplotLayer',
                    data=tr,
                    get_position='[center_lon, center_lat]',
                    get_radius='100 + total_crashes * 30',
                    get_fill_color='[fill_r, fill_g, fill_b]',
                    pickable=True,
                    opacity=0.7,
                )
                deck = pdk.Deck(layers=[scatter], initial_view_state=initial,
                                tooltip={'text': 'Cell: {h3_cell}\nTrend: {trend_direction}\nSignificance: {trend_significance}\nTotal crashes: {total_crashes}'})
                st.pydeck_chart(deck, use_container_width=True)

        with c2:
            # Distribution of trend slopes
            if 'trend_slope' in tr.columns:
                st.markdown('**Distribution of Trend Slopes**')
                slope_chart = alt.Chart(tr.head(10000)).mark_bar().encode(
                    x=alt.X('trend_slope:Q', bin=alt.Bin(maxbins=30), title='Trend slope'),
                    y=alt.Y('count()', title='Count'),
                    tooltip=[alt.Tooltip('count()', title='Count')]
                ).properties(height=280)
                st.altair_chart(slope_chart, use_container_width=True)

            # Trend direction counts
            if 'trend_direction' in tr.columns:
                st.markdown('**Trend Direction Breakdown**')
                dir_chart = alt.Chart(tr).mark_bar().encode(
                    x=alt.X('trend_direction:N', title='Direction'),
                    y=alt.Y('count()', title='Cells'),
                    color='trend_direction:N',
                    tooltip=[alt.Tooltip('count()', title='Cells')]
                ).properties(height=260)
                st.altair_chart(dir_chart, use_container_width=True)

# --- Tab 3: Alarms ---
with tab3:
    st.subheader('Spatiotemporal Alarms')
    if al is None or al.empty:
        st.info('No alarm data loaded.')
    else:
        map_center = center_on(al, 'center_lat', 'center_lon', default=(40.72, -74.0))
        initial = pdk.ViewState(latitude=map_center[0], longitude=map_center[1], zoom=10, pitch=30)

        # Color by alarm type
        alarm_palette = {
            'New Hotspot': (230, 120, 80),
            "Intensifying Hotspot": (230, 80, 160),
            'Persistent Hotspot': (120, 120, 240),
        }
        if 'alarm_type' in al.columns:
            al['_r'] = al['alarm_type'].map(lambda x: alarm_palette.get(x, (190, 190, 190))[0])
            al['_g'] = al['alarm_type'].map(lambda x: alarm_palette.get(x, (190, 190, 190))[1])
            al['_b'] = al['alarm_type'].map(lambda x: alarm_palette.get(x, (190, 190, 190))[2])
        else:
            al[['_r','_g','_b']] = (190,190,190)

        layer = pdk.Layer(
            'ScatterplotLayer',
            data=al,
            get_position='[center_lon, center_lat]',
            get_radius='250 + crash_count * 100',
            get_fill_color='[_r, _g, _b]',
            pickable=True,
            opacity=0.75,
        )
        deck = pdk.Deck(layers=[layer], initial_view_state=initial,
                        tooltip={'text': 'Cell: {h3_cell}\nType: {alarm_type}\nCrashes: {crash_count}\nGi*: {gi_star}\nPriority: {priority}\nAction: {recommended_action}'})
        st.pydeck_chart(deck, use_container_width=True)

        st.markdown('### Alarm Details')
        show_cols = [c for c in ['time_bin', 'alarm_type', 'h3_cell', 'crash_count', 'gi_star', 'priority', 'recommended_action', 'hot_period_count', 'center_lat', 'center_lon'] if c in al.columns]
        st.dataframe(al[show_cols].head(max_rows), use_container_width=True)

        # Download filtered alarms
        csv = al[show_cols].to_csv(index=False).encode('utf-8')
        st.download_button('Download filtered alarms (CSV)', data=csv, file_name='filtered_alarms.csv', mime='text/csv')

# --- Tab 4: Overview ---
with tab4:
    st.subheader('Overview & Quick Cuts')
    c1, c2 = st.columns(2, gap='large')

    with c1:
        if hs is not None and not hs.empty and 'time_bin' in hs.columns and 'crash_count' in hs.columns:
            st.markdown('**Crashes by Time Period (Hotspots)**')
            tmp = hs.groupby('time_bin', as_index=False)['crash_count'].sum()
            tmp['time_order'] = as_time_order(tmp['time_bin'])
            chart = alt.Chart(tmp).mark_bar().encode(
                x=alt.X('time_bin:N', sort=list(tmp.sort_values('time_order')['time_bin']), title='Time period'),
                y=alt.Y('crash_count:Q', title='Total crashes'),
                tooltip=['time_bin:N', 'crash_count:Q']
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

    with c2:
        if tr is not None and not tr.empty and 'trend_direction' in tr.columns and 'total_crashes' in tr.columns:
            st.markdown('**Total Crashes by Trend Direction**')
            tmp = tr.groupby('trend_direction', as_index=False)['total_crashes'].sum()
            chart = alt.Chart(tmp).mark_arc(innerRadius=60).encode(
                theta='total_crashes:Q',
                color='trend_direction:N',
                tooltip=['trend_direction:N', 'total_crashes:Q']
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

# --- Tab 5: Raw Data ---
with tab5:
    st.subheader('Raw Data')
    st.caption('Preview up to the configured max rows. Use the download buttons to export filtered snapshots.')

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Hotspots**')
        if hs is None or hs.empty:
            st.write('—')
        else:
            st.dataframe(hs.head(max_rows), use_container_width=True)
            csv = hs.to_csv(index=False).encode('utf-8')
            st.download_button('Download hotspots (CSV)', data=csv, file_name='hotspots_filtered.csv', mime='text/csv')

    with c2:
        st.markdown('**Trends**')
        if tr is None or tr.empty:
            st.write('—')
        else:
            st.dataframe(tr.head(max_rows), use_container_width=True)
            csv = tr.to_csv(index=False).encode('utf-8')
            st.download_button('Download trends (CSV)', data=csv, file_name='trends_filtered.csv', mime='text/csv')

    with c3:
        st.markdown('**Alarms**')
        if al is None or al.empty:
            st.write('—')
        else:
            st.dataframe(al.head(max_rows), use_container_width=True)
            csv = al.to_csv(index=False).encode('utf-8')
            st.download_button('Download alarms (CSV)', data=csv, file_name='alarms_filtered.csv', mime='text/csv')


# --------------------------
# Footer
# --------------------------
with st.expander('ℹ️ Notes & Tips', expanded=False):
    st.markdown(
        '''
        - This app uses **pydeck's H3HexagonLayer** in the browser, so you don't need the Python `h3` package.
        - If H3 rendering fails in your environment, it falls back to a **scatter map** using cell centroids.
        - Use the sidebar to **upload CSVs** or pre-place them in the same folder as this script.
        - For Streamlit Cloud, create a `requirements.txt` with `streamlit`, `pandas`, `pydeck`, and `altair`.
        '''
    )
