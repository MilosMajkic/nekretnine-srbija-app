import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from folium.plugins import MarkerCluster, HeatMap, Draw
from streamlit_folium import st_folium
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor

# ────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate PRO",
    layout="wide",
    page_icon="🏢"
)

CITY_CENTERS = {
    "Belgrade": (44.8150, 20.4612),
    "Novi Sad": (45.2516, 19.8369),
    "Niš": (43.3209, 21.8957),
    "Kragujevac": (44.0128, 20.9114)
}

# Initialize favorites
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []

# ────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@st.cache_data
def load_local_data(mode):
    if mode == "Buy Apartments":
        file_path = "nekretnina_srb_geo.csv"
    else:
        file_path = "izdavanje_srb_geo.csv"

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        st.error(f"File **{file_path}** not found in the application folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

    # Expected column mapping (Serbian → English)
    expected_cols = {
        'grad': 'City',
        'deoGrada': 'District',
        'ulica': 'Street',
        'cenaEur': 'Price_EUR',
        'kvadraturaM2': 'Area_m2',
        'cenaPoM2': 'Price_per_m2',
        'naslov': 'Title',
        'link': 'Link',
        'latitude': 'Latitude',
        'longitude': 'Longitude'
    }

    # Only rename columns that actually exist
    rename_dict = {k: v for k, v in expected_cols.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    # Convert to numeric where needed
    numeric_cols = ['Latitude', 'Longitude', 'Price_EUR', 'Area_m2', 'Price_per_m2']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ──────────────── DEBUG INFO ────────────────
    st.subheader("CSV Debug Info (remove in production if you want)")
    st.write("**Columns present after loading & renaming:**")
    st.write(df.columns.tolist())

    missing_expected = [v for v in expected_cols.values() if v not in df.columns]
    if missing_expected:
        st.warning(f"These expected columns are **missing**: {', '.join(missing_expected)}")
        st.info("Check your CSV file headers. They must match one of: " + ", ".join(expected_cols.keys()))

    if len(df) > 0:
        st.write("**First 3 rows (sample):**")
        st.dataframe(df.head(3))
    # ────────────────────────────────────────────

    return df


@st.cache_resource
def train_model(data):
    required = {'Area_m2', 'Latitude', 'Longitude', 'Price_EUR'}
    available = set(data.columns)

    missing = required - available
    if missing:
        st.error(f"Cannot train price prediction model — missing columns: {', '.join(missing)}")
        return None

    subset_df = data[list(required)]
    valid_data = subset_df.dropna()

    if len(valid_data) < 10:
        st.warning("Too few complete rows with Area, Lat, Lon and Price (< 10). Model not trained.")
        return None

    X = valid_data[['Area_m2', 'Latitude', 'Longitude']]
    y = valid_data['Price_EUR']

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


# ────────────────────────────────────────────────
# MAIN APPLICATION
# ────────────────────────────────────────────────
st.title("Real Estate Market Analytics (Local CSV)")

# ── SIDEBAR ──
st.sidebar.header("Settings")
mode = st.sidebar.radio("Market:", ["Buy Apartments", "Rent Apartments"])

with st.spinner("Loading local data..."):
    df = load_local_data(mode)

if df.empty:
    st.error("No data loaded. Check file existence and format.")
    st.stop()

with st.spinner("Preparing data, training model and calculating features..."):
    model = train_model(df)

    # Default: predicted = actual (if no model)
    df['Predicted_Price'] = df['Price_EUR']

    if model is not None:
        mask = df[['Latitude', 'Longitude', 'Area_m2']].notna().all(axis=1)
        if mask.any():
            preds = model.predict(df.loc[mask, ['Area_m2', 'Latitude', 'Longitude']])
            df.loc[mask, 'Predicted_Price'] = preds

    df['AI_Ratio'] = df['Price_EUR'] / df['Predicted_Price'].replace(0, np.nan)
    df['Good_Deal'] = df['AI_Ratio'] <= 0.85
    df['Suspiciously_Low'] = df['AI_Ratio'] <= 0.45

    df['Distance_to_Center_km'] = df.apply(
        lambda row: haversine_distance(
            row.get('Latitude'), row.get('Longitude'),
            CITY_CENTERS.get(row.get('City'), (np.nan, np.nan))[0],
            CITY_CENTERS.get(row.get('City'), (np.nan, np.nan))[1]
        ) if row.get('City') in CITY_CENTERS else None,
        axis=1
    )

    df['Walk_Score'] = df['Distance_to_Center_km'].apply(
        lambda x: max(0, min(100, int(100 - (x or 0) * 8))) if pd.notna(x) else 50
    )

# ── SIDEBAR FILTERS ──
st.sidebar.markdown("---")
view = st.sidebar.radio("Main Menu:", [
    "Interactive Map",
    "Analytics & Mortgage",
    "Location A/B Comparison",
    "Favorites"
])

st.sidebar.markdown("---")
st.sidebar.header("Filters")

cities = ["All"] + sorted(df['City'].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("City:", cities)

price_label = "Price (EUR)" if mode == "Buy Apartments" else "Monthly Rent (EUR)"
max_price_default = 300000 if mode == "Buy Apartments" else 2000

price_range = st.sidebar.slider(
    price_label,
    int(df['Price_EUR'].min(skipna=True) or 0),
    int(df['Price_EUR'].max(skipna=True) or max_price_default),
    (int(df['Price_EUR'].min(skipna=True) or 0), int(df['Price_EUR'].max(skipna=True) or max_price_default))
)

price_m2_range = st.sidebar.slider(
    "Price per m² (EUR):",
    int(df['Price_per_m2'].min(skipna=True) or 0),
    int(df['Price_per_m2'].max(skipna=True) or 5000),
    (int(df['Price_per_m2'].min(skipna=True) or 0), int(df['Price_per_m2'].max(skipna=True) or 5000))
)

area_range = st.sidebar.slider(
    "Area (m²):",
    int(df['Area_m2'].min(skipna=True) or 10),
    int(df['Area_m2'].max(skipna=True) or 200),
    (int(df['Area_m2'].min(skipna=True) or 10), int(df['Area_m2'].max(skipna=True) or 200))
)

if selected_city in CITY_CENTERS:
    max_dist = math.ceil(df[df['City'] == selected_city]['Distance_to_Center_km'].max() or 20)
    distance_range = st.sidebar.slider("Distance from center (km):", 0, max_dist, max_dist)
else:
    distance_range = 999

only_good_deals = st.sidebar.checkbox("Show only 'Great Deals'")
show_heatmap = st.sidebar.checkbox("Show Heatmap")
show_3d = st.sidebar.checkbox("Show 3D Density (PyDeck)")

# ── FILTERING ──
filtered_df = df[
    (df['Price_EUR'] >= price_range[0]) & (df['Price_EUR'] <= price_range[1]) &
    (df['Price_per_m2'] >= price_m2_range[0]) & (df['Price_per_m2'] <= price_m2_range[1]) &
    (df['Area_m2'] >= area_range[0]) & (df['Area_m2'] <= area_range[1])
]

if selected_city != "All":
    filtered_df = filtered_df[filtered_df['City'] == selected_city]
    if selected_city in CITY_CENTERS:
        filtered_df = filtered_df[filtered_df['Distance_to_Center_km'] <= distance_range]

if only_good_deals:
    filtered_df = filtered_df[filtered_df['Good_Deal']]

filtered_df = filtered_df[~filtered_df['Suspiciously_Low']]

map_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])

# ────────────────────────────────────────────────
# VIEW SELECTION
# ────────────────────────────────────────────────
if view == "Interactive Map":
    st.header("Geospatial Analysis")
    st.markdown(f"**Found {len(filtered_df):,} listings**   Average {price_label.lower()}: **{filtered_df['Price_EUR'].mean():,.0f} €**")

    if len(map_df) == 0:
        st.warning("No listings with valid coordinates after filtering.")
    else:
        # Limit markers for performance
        if len(map_df) > 1000:
            display_df = map_df.sample(n=1000, random_state=42)
            st.info(f"Displaying **1000** random listings out of {len(map_df):,} (performance reasons).")
        else:
            display_df = map_df

        center_lat = display_df['Latitude'].median()
        center_lon = display_df['Longitude'].median()

        if show_3d:
            st.subheader("3D Hexagon Density")
            layer = pdk.Layer(
                'HexagonLayer',
                data=display_df,
                get_position='[Longitude, Latitude]',
                radius=150,
                elevation_scale=15,
                elevation_range=[0, 2000],
                extruded=True,
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=50)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))
        else:
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")
            Draw(export=True).add_to(m)

            if show_heatmap:
                heat_data = [[r['Latitude'], r['Longitude'], r['Price_per_m2']] for _, r in display_df.iterrows()]
                HeatMap(heat_data, radius=14, blur=20).add_to(m)

            marker_cluster = MarkerCluster().add_to(m)
            for _, row in display_df.iterrows():
                color = "green" if row['Good_Deal'] else "blue"
                icon_name = "star" if row['Good_Deal'] else "home"
                status_text = "<b style='color:green;'>Great Deal!</b>" if row['Good_Deal'] else "Fair price"

                popup_html = f"""
                <div style="width:260px; font-size:13px;">
                    <b>{row.get('Title', 'No title')}</b><br>
                    <small>{row.get('Street', '')} • {row.get('District', '')}</small><hr>
                    <b>Price:</b> {row['Price_EUR']:,.0f} €<br>
                    <b>{row['Area_m2']}</b> m²  •  <b>{row.get('Price_per_m2', '–'):,.0f} €/m²</b><br>
                    Walk Score: <b>{row.get('Walk_Score', 50)}</b>/100<br>
                    <small>AI estimate ≈ {row['Predicted_Price']:,.0f} €</small><br>
                    {status_text}<br>
                    <a href="{row.get('Link', '#')}" target="_blank" style="display:block; margin-top:8px; background:#0066cc; color:white; text-align:center; padding:6px; border-radius:4px; text-decoration:none;">Open listing</a>
                </div>
                """

                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(popup_html, max_width=320),
                    icon=folium.Icon(color=color, icon=icon_name, prefix="fa")
                ).add_to(marker_cluster)

            st_folium(m, width="100%", height=600)

        # Quick view table + favorites
        st.subheader("Quick view + add to favorites")
        cols_to_show = ['Title', 'Price_EUR', 'Area_m2', 'Price_per_m2', 'District', 'Link']
        display_table = filtered_df[[c for c in cols_to_show if c in filtered_df.columns]].copy()
        display_table.insert(0, "Save", False)

        edited_df = st.data_editor(
            display_table,
            column_config={"Save": st.column_config.CheckboxColumn(required=True)},
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )

        if st.button("Save selected to Favorites"):
            new_items = edited_df[edited_df["Save"]].drop(columns="Save").to_dict("records")
            added_count = 0
            existing_links = {x['Link'] for x in st.session_state['favorites'] if 'Link' in x}
            for item in new_items:
                if item.get('Link') and item['Link'] not in existing_links:
                    st.session_state['favorites'].append(item)
                    added_count += 1
            if added_count > 0:
                st.success(f"Added **{added_count}** new listings to favorites!")
            else:
                st.info("No new items added (already in favorites or no link).")

elif view == "Analytics & Mortgage":
    st.info("Analytics & Mortgage section – to be implemented")

elif view == "Location A/B Comparison":
    st.info("Location A/B Comparison – to be implemented")

elif view == "Favorites":
    st.info("Favorites view – to be implemented")

else:
    st.info("Select a view from the menu on the left.")
