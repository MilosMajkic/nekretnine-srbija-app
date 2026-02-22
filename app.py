import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from folium.plugins import MarkerCluster, HeatMap, Draw
from streamlit_folium import st_folium
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor

# --- KONFIGURACIJA ---
st.set_page_config(page_title="Nekretnine PRO", layout="wide", page_icon="🏢")

CENTRI_GRADOVA = {
    "Beograd": (44.8150, 20.4612),
    "Novi Sad": (45.2516, 19.8369),
    "Niš": (43.3209, 21.8957),
    "Kragujevac": (44.0128, 20.9114)
}

# Inicijalizacija omiljenih
if 'omiljeni' not in st.session_state:
    st.session_state['omiljeni'] = []

# --- POMOĆNE FUNKCIJE ---
def haversine_distance(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return None
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@st.cache_data
def load_local_data(rezim_rada):
    if rezim_rada == "Kupovina stanova":
        file_path = "nekretnina_srb_geo.csv"
    else:
        file_path = "izdavanje_srb_geo.csv"
    
    try:
        df = pd.read_csv(file_path, low_memory=False)
        
        # Pretpostavljeni nazivi kolona – prilagodi ako su drugačiji!
        expected_cols = {
            'grad': 'Grad',
            'deoGrada': 'Deo_Grada',
            'ulica': 'Ulica',
            'cenaEur': 'Cena_EUR',
            'kvadraturaM2': 'Kvadratura_m2',
            'cenaPoM2': 'Cena_po_m2',
            'naslov': 'Naslov',
            'link': 'Link',
            'latitude': 'Latitude',
            'longitude': 'Longitude'
        }
        
        # Preimenujemo samo one kolone koje postoje
        rename_dict = {k: v for k, v in expected_cols.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Osiguravamo numeričke kolone
        for col in ['Latitude', 'Longitude', 'Cena_EUR', 'Kvadratura_m2', 'Cena_po_m2']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except FileNotFoundError:
        st.error(f" fajl **{file_path}** nije pronađen u folderu aplikacije.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Greška pri učitavanju fajla: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_model(data):
    valid_data = data.dropna(subset=['Kvadratura_m2', 'Latitude', 'Longitude', 'Cena_EUR'])
    if len(valid_data) < 10:
        return None
    
    X = valid_data[['Kvadratura_m2', 'Latitude', 'Longitude']]
    y = valid_data['Cena_EUR']
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model

# ────────────────────────────────────────────────
#               GLAVNI DEO APLIKACIJE
# ────────────────────────────────────────────────

st.title("Analitika Tržišta Nekretnina (lokalni CSV)")

# SIDEBAR
st.sidebar.header("⚙️ Podešavanja")
rezim = st.sidebar.radio("Tržište:", ["Kupovina stanova", "Izdavanje stanova"])

with st.spinner("Učitavam lokalne podatke..."):
    df = load_local_data(rezim)

if len(df) == 0:
    st.stop()

with st.spinner("Treniram model i računam distance..."):
    model = train_model(df)
    
    df['Predvidjena_Cena'] = df['Cena_EUR']
    if model is not None:
        mask = df[['Latitude', 'Longitude']].notna().all(axis=1)
        if mask.any():
            df.loc[mask, 'Predvidjena_Cena'] = model.predict(
                df.loc[mask, ['Kvadratura_m2', 'Latitude', 'Longitude']]
            )
    
    df['Odnos_AI'] = df['Cena_EUR'] / df['Predvidjena_Cena'].replace(0, np.nan)
    df['Dobra_Prilika'] = df['Odnos_AI'] <= 0.85
    df['Sumnjivo_Nisko'] = df['Odnos_AI'] <= 0.45
    
    df['Udaljenost_Centar_km'] = df.apply(
        lambda row: haversine_distance(
            row['Latitude'], row['Longitude'],
            CENTRI_GRADOVA.get(row['Grad'], (np.nan, np.nan))[0],
            CENTRI_GRADOVA.get(row['Grad'], (np.nan, np.nan))[1]
        ) if row['Grad'] in CENTRI_GRADOVA else None,
        axis=1
    )
    
    df['Walk_Score'] = df['Udaljenost_Centar_km'].apply(
        lambda x: max(0, min(100, int(100 - (x or 0) * 8))) if pd.notna(x) else 50
    )

# ────────────── SIDEBAR FILTERI ──────────────
st.sidebar.markdown("---")
prikaz = st.sidebar.radio("🧭 Glavni meni:", [
    "📍 Interaktivna Mapa",
    "📈 Analitika & Krediti",
    "⚖️ A/B Poređenje Lokacija",
    "⭐ Omiljeni Oglasi",
])

st.sidebar.markdown("---")
st.sidebar.header("🔍 Filteri")

gradovi = ["Svi"] + sorted(df['Grad'].dropna().unique().tolist())
izabrani_grad = st.sidebar.selectbox("Grad:", gradovi)

label_cena = "Cena (EUR)" if rezim == "Kupovina stanova" else "Mesečna kirija (EUR)"
max_c = 300000 if rezim == "Kupovina stanova" else 2000

cena_opseg = st.sidebar.slider(
    label_cena,
    int(df['Cena_EUR'].min(skipna=True) or 0),
    int(df['Cena_EUR'].max(skipna=True) or max_c),
    (int(df['Cena_EUR'].min(skipna=True) or 0), int(df['Cena_EUR'].max(skipna=True) or max_c))
)

m2_opseg = st.sidebar.slider(
    "Cena po m² (EUR):",
    int(df['Cena_po_m2'].min(skipna=True) or 0),
    int(df['Cena_po_m2'].max(skipna=True) or 5000),
    (int(df['Cena_po_m2'].min(skipna=True) or 0), int(df['Cena_po_m2'].max(skipna=True) or 5000))
)

kv_opseg = st.sidebar.slider(
    "Kvadratura (m²):",
    int(df['Kvadratura_m2'].min(skipna=True) or 10),
    int(df['Kvadratura_m2'].max(skipna=True) or 200),
    (int(df['Kvadratura_m2'].min(skipna=True) or 10), int(df['Kvadratura_m2'].max(skipna=True) or 200))
)

if izabrani_grad in CENTRI_GRADOVA:
    max_dist = math.ceil(df[df['Grad'] == izabrani_grad]['Udaljenost_Centar_km'].max() or 20)
    dist_opseg = st.sidebar.slider("Udaljenost od centra (km):", 0, max_dist, max_dist)
else:
    dist_opseg = 999

samo_prilike = st.sidebar.checkbox("🌟 Samo 'Zlatne Prilike'")
prikazi_heatmap = st.sidebar.checkbox("🔥 Toplotna mapa")
prikazi_3d = st.sidebar.checkbox("🏙️ 3D gustina (PyDeck)")

# FILTRIRANJE
f_df = df[
    (df['Cena_EUR'] >= cena_opseg[0]) & (df['Cena_EUR'] <= cena_opseg[1]) &
    (df['Cena_po_m2'] >= m2_opseg[0]) & (df['Cena_po_m2'] <= m2_opseg[1]) &
    (df['Kvadratura_m2'] >= kv_opseg[0]) & (df['Kvadratura_m2'] <= kv_opseg[1])
]

if izabrani_grad != "Svi":
    f_df = f_df[f_df['Grad'] == izabrani_grad]
    if izabrani_grad in CENTRI_GRADOVA:
        f_df = f_df[f_df['Udaljenost_Centar_km'] <= dist_opseg]

if samo_prilike:
    f_df = f_df[f_df['Dobra_Prilika']]

f_df = f_df[~f_df['Sumnjivo_Nisko']]

map_df = f_df.dropna(subset=['Latitude', 'Longitude'])

# ────────────── PRIKAZ PO MODOVIMA ──────────────

if prikaz == "📍 Interaktivna Mapa":
    st.header("Geoprostorna analiza")
    st.markdown(f"**Pronađeno ukupno {len(f_df):,} oglasa**    Prosečna {label_cena.lower()}: **{f_df['Cena_EUR'].mean():,.0f} €**")

    if len(map_df) == 0:
        st.warning("Nema oglasa sa ispravnim koordinatama nakon filtera.")
    else:
        # Limit prikaza markera
        if len(map_df) > 1000:
            prikaz_df = map_df.sample(n=1000, random_state=42)   # ili .head(1000)
            st.info(f"Prikazano **1000** nasumičnih oglasa od ukupno {len(map_df):,} (zbog performansi). Svi podaci su u tabeli ispod.")
        else:
            prikaz_df = map_df

        start_lat = prikaz_df['Latitude'].median()
        start_lon = prikaz_df['Longitude'].median()

        if prikazi_3d:
            st.subheader("3D Hexagon gustina")
            layer = pdk.Layer(
                'HexagonLayer',
                data=prikaz_df,
                get_position='[Longitude, Latitude]',
                radius=150,
                elevation_scale=15,
                elevation_range=[0, 2000],
                extruded=True,
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=start_lat, longitude=start_lon, zoom=11, pitch=50)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

        else:
            m = folium.Map(location=[start_lat, start_lon], zoom_start=12, tiles="CartoDB positron")
            Draw(export=True).add_to(m)

            if prikazi_heatmap:
                heat_data = [[r['Latitude'], r['Longitude'], r['Cena_po_m2']] for _, r in prikaz_df.iterrows()]
                HeatMap(heat_data, radius=14, blur=20).add_to(m)

            marker_cluster = MarkerCluster().add_to(m)

            for _, row in prikaz_df.iterrows():
                boja = "green" if row['Dobra_Prilika'] else "blue"
                ikona = "star" if row['Dobra_Prilika'] else "home"
                status = "<b style='color:green;'>🌟 Odlična prilika!</b>" if row['Dobra_Prilika'] else "Realna cena"

                html = f"""
                <div style="width:260px; font-size:13px;">
                    <b>{row['Naslov']}</b><br>
                    <small>{row.get('Ulica','')} • {row['Deo_Grada']}</small><hr>
                    <b>Cena:</b> {row['Cena_EUR']:,.0f} €<br>
                    <b>{row['Kvadratura_m2']}</b> m²  •  <b>{row['Cena_po_m2']:,.0f} €/m²</b><br>
                    Walk Score: <b>{row.get('Walk_Score',50)}</b>/100<br>
                    <small>AI procena ≈ {row['Predvidjena_Cena']:,.0f} €</small><br>
                    {status}<br>
                    <a href="{row['Link']}" target="_blank" style="display:block; margin-top:8px; background:#0066cc; color:white; text-align:center; padding:6px; border-radius:4px; text-decoration:none;">Otvori oglas</a>
                </div>
                """

                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=folium.Popup(html, max_width=320),
                    icon=folium.Icon(color=boja, icon=ikona, prefix="fa")
                ).add_to(marker_cluster)

            st_folium(m, width="100%", height=600)

        # Tabela za čuvanje
        st.subheader("Brzi pregled + dodaj u omiljene")
        df_prikaz = f_df[['Naslov', 'Cena_EUR', 'Kvadratura_m2', 'Cena_po_m2', 'Deo_Grada', 'Link']].copy()
        df_prikaz.insert(0, "Sačuvaj", False)

        edited = st.data_editor(
            df_prikaz,
            column_config={"Sačuvaj": st.column_config.CheckboxColumn(required=True)},
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )

        if st.button("Sačuvaj označene u Omiljene"):
            novi = edited[edited["Sačuvaj"]].drop(columns="Sačuvaj").to_dict("records")
            added = 0
            for oglas in novi:
                if oglas['Link'] not in {x['Link'] for x in st.session_state['omiljeni']}:
                    st.session_state['omiljeni'].append(oglas)
                    added += 1
            if added > 0:
                st.success(f"Dodato **{added}** novih oglasa u omiljene!")
            else:
                st.info("Svi označeni su već u omiljenima.")

# Ostali prikazi (Analitika, A/B, Omiljeni) ostaju isti kao u originalu...
# (možeš ih kopirati iz prethodne verzije – izmene su uglavnom bile u učitavanju i mapiranju)

elif prikaz == "📈 Analitika & Krediti":
    # ... tvoj originalni kod za ovaj deo ...

elif prikaz == "⚖️ A/B Poređenje Lokacija":
    # ... tvoj originalni kod ...

elif prikaz == "⭐ Omiljeni Oglasi":
    # ... tvoj originalni kod ...

else:
    st.info("Odaberi prikaz iz menija levo.")
