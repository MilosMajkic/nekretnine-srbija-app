import streamlit as st
import requests
import pandas as pd
import numpy as np
import math
import folium
from folium.plugins import MarkerCluster, HeatMap, Draw
from streamlit_folium import st_folium
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor
import json

# --- KONFIGURACIJA ---
st.set_page_config(page_title="Nekretnine PRO", layout="wide", page_icon="🏢")

CENTRI_GRADOVA = {
    "Beograd": (44.8150, 20.4612),
    "Novi Sad": (45.2516, 19.8369),
    "Niš": (43.3209, 21.8957),
    "Kragujevac": (44.0128, 20.9114)
}

# Inicijalizacija Session State-a za "Omiljene" oglase
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

# 🛑 SPECIJALNO UČITAVANJE - Zaobilazi Somee reklamu!
@st.cache_data(ttl=300)
def load_data(rezim_rada):
    api_url = "http://bekend.somee.com/api/nekretnine"
    try:
        odgovor = requests.get(api_url, params={"rezim": rezim_rada})
        if odgovor.status_code == 200:
            
            # --- HAK ZA SOMEE: Sečemo samo ono što je između [ i ] ---
            tekst = odgovor.text
            pocetak = tekst.find('[')
            kraj = tekst.rfind(']')
            
            if pocetak != -1 and kraj != -1:
                cist_json = tekst[pocetak:kraj+1]
                podaci = json.loads(cist_json)
                
                if len(podaci) > 0:
                    df = pd.DataFrame(podaci)
                    
                    # Prevodimo kolone sa C# na tvoj Python kod
                    mapiranje = {
                        'grad': 'Grad', 'deoGrada': 'Deo_Grada', 'ulica': 'Ulica',
                        'cenaEur': 'Cena_EUR', 'kvadraturaM2': 'Kvadratura_m2', 
                        'cenaPoM2': 'Cena_po_m2', 'naslov': 'Naslov', 'link': 'Link',
                        'latitude': 'Latitude', 'longitude': 'Longitude'
                    }
                    df = df.rename(columns=mapiranje)
                    
                    # Osiguravamo da su koordinate brojevi (da mapa ne pukne)
                    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                    
                    # Ne brišemo prazne koordinate još uvek, da bismo videli podatke!
                    # df = df.dropna(subset=['Latitude', 'Longitude', 'Cena_EUR', 'Kvadratura_m2'])
                    return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Ne mogu da se povežem sa serverom. Detalji: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_model(data):
    # Proveravamo da li imamo dovoljno ispravnih podataka za ML
    valid_data = data.dropna(subset=['Kvadratura_m2', 'Latitude', 'Longitude', 'Cena_EUR'])
    if len(valid_data) == 0: return None
    
    X = valid_data[['Kvadratura_m2', 'Latitude', 'Longitude']]
    y = valid_data['Cena_EUR']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

# --- APLIKACIJA ---
st.title("🏢 Premium Analitika Tržišta Nekretnina")

# SIDEBAR: REŽIM I FILTERI
st.sidebar.header("⚙️ Podešavanja")
rezim = st.sidebar.radio("Tržište:", ["Kupovina stanova", "Izdavanje stanova"])
df = load_data(rezim)

if len(df) > 0:
    with st.spinner('Učitavam AI algoritam i mapiram lokacije...'):
        model = train_model(df)
        
        # AI Procena (samo za redove koji imaju koordinate)
        df['Predvidjena_Cena'] = df['Cena_EUR'] # Default
        if model is not None:
            mask = df['Latitude'].notna() & df['Longitude'].notna()
            if mask.any():
                df.loc[mask, 'Predvidjena_Cena'] = model.predict(df.loc[mask, ['Kvadratura_m2', 'Latitude', 'Longitude']])

        df['Odnos_AI'] = df['Cena_EUR'] / df['Predvidjena_Cena']
        df['Dobra_Prilika'] = df['Odnos_AI'] <= 0.85
        df['Sumnjivo_Nisko'] = df['Odnos_AI'] <= 0.45

        df['Udaljenost_Centar_km'] = df.apply(
            lambda row: haversine_distance(row['Latitude'], row['Longitude'], CENTRI_GRADOVA[row['Grad']][0], CENTRI_GRADOVA[row['Grad']][1])
            if row.get('Grad') in CENTRI_GRADOVA else None, axis=1
        )
        df['Walk_Score'] = df['Udaljenost_Centar_km'].apply(
            lambda x: max(0, min(100, int(100 - (x * 8)))) if pd.notna(x) else 50)

    # --- GLAVNI MENI ---
    st.sidebar.markdown("---")
    prikaz = st.sidebar.radio("🧭 Glavni meni:", [
        "📍 Interaktivna Mapa",
        "📈 Analitika & Krediti",
        "⚖️ A/B Poređenje Lokacija",
        "⭐ Omiljeni Oglasi",
        "⚙️ Integracije (Webhook)"
    ])
    st.sidebar.markdown("---")

    st.sidebar.header("🔍 Filteri")
    gradovi = ["Svi"] + sorted(df['Grad'].dropna().unique().tolist())
    izabrani_grad = st.sidebar.selectbox("Grad:", gradovi)

    label_cena = "Cena (EUR)" if rezim == "Kupovina stanova" else "Mesečna Kirija (EUR)"
    max_c = 300000 if rezim == "Kupovina stanova" else 2000
    
    min_cena_val = int(df['Cena_EUR'].min()) if not df['Cena_EUR'].empty and not pd.isna(df['Cena_EUR'].min()) else 0
    max_cena_val = int(df['Cena_EUR'].max()) if not df['Cena_EUR'].empty and not pd.isna(df['Cena_EUR'].max()) else max_c
    cena_opseg = st.sidebar.slider(label_cena, min_cena_val, max_cena_val, (min_cena_val, max_cena_val))

    min_m2 = int(df['Cena_po_m2'].min()) if not pd.isna(df['Cena_po_m2'].min()) else 0
    max_m2 = int(df['Cena_po_m2'].max()) if not pd.isna(df['Cena_po_m2'].max()) else 5000
    m2_opseg = st.sidebar.slider("Cena po m² (EUR):", min_m2, max_m2, (min_m2, max_m2))

    min_kv = int(df['Kvadratura_m2'].min()) if not pd.isna(df['Kvadratura_m2'].min()) else 10
    max_kv = int(df['Kvadratura_m2'].max()) if not pd.isna(df['Kvadratura_m2'].max()) else 200
    kv_opseg = st.sidebar.slider("Kvadratura (m²):", min_kv, max_kv, (min_kv, max_kv))

    if izabrani_grad in CENTRI_GRADOVA:
        max_dist = math.ceil(df[df['Grad'] == izabrani_grad]['Udaljenost_Centar_km'].max() or 20)
        dist_opseg = st.sidebar.slider("Udaljenost od strogog centra (km):", 0, max_dist, max_dist)
    else:
        dist_opseg = 999

    samo_prilike = st.sidebar.checkbox("🌟 Samo 'Zlatne Prilike'")
    prikazi_heatmap = st.sidebar.checkbox("🔥 Toplotna Mapa (2D)")
    prikazi_3d = st.sidebar.checkbox("🏙️ Prikaži 3D Gustinu (PyDeck)")

    # Filtriranje
    f_df = df[
        (df['Cena_EUR'] >= cena_opseg[0]) & (df['Cena_EUR'] <= cena_opseg[1]) &
        (df['Cena_po_m2'] >= m2_opseg[0]) & (df['Cena_po_m2'] <= m2_opseg[1]) &
        (df['Kvadratura_m2'] >= kv_opseg[0]) & (df['Kvadratura_m2'] <= kv_opseg[1])
    ]
    if izabrani_grad != "Svi":
        f_df = f_df[f_df['Grad'] == izabrani_grad]
        if izabrani_grad in CENTRI_GRADOVA:
            f_df = f_df[f_df['Udaljenost_Centar_km'] <= dist_opseg]

    if samo_prilike: f_df = f_df[f_df['Dobra_Prilika'] == True]
    f_df = f_df[f_df['Sumnjivo_Nisko'] == False]

    # Zadržavamo samo redove sa validnim koordinatama za mapu
    map_df = f_df.dropna(subset=['Latitude', 'Longitude'])

    # ---------------- PRIKAZ 1: MAPA ----------------
    if prikaz == "📍 Interaktivna Mapa":
        st.header("Geoprostorna analiza")
        st.markdown(f"**Prikazujem {len(f_df)} oglasa.** Prosečna {label_cena.lower()}: **{f_df['Cena_EUR'].mean():,.0f} €**")

        if len(map_df) > 0:
            start_coords = [map_df['Latitude'].iloc[0], map_df['Longitude'].iloc[0]]

            if prikazi_3d:
                st.subheader("🏙️ 3D Mapa Gustine Tržišta")
                layer = pdk.Layer(
                    'HexagonLayer',
                    data=map_df,
                    get_position='[Longitude, Latitude]',
                    radius=200,
                    elevation_scale=10,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                )
                view_state = pdk.ViewState(latitude=start_coords[0], longitude=start_coords[1], zoom=12, pitch=50)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Koncentracija stanova"}))
            else:
                m = folium.Map(location=start_coords, zoom_start=12, tiles="OpenStreetMap")
                Draw(export=True).add_to(m)

                if prikazi_heatmap:
                    heat_data = [[r['Latitude'], r['Longitude'], r['Cena_po_m2']] for i, r in map_df.iterrows()]
                    HeatMap(heat_data, radius=15).add_to(m)

                marker_cluster = MarkerCluster().add_to(m)

                for index, row in map_df.iterrows():
                    boja, ikona = "blue", "home"
                    status = "Realna cena"
                    if row['Dobra_Prilika']: boja, ikona, status = "green", "star", "<b style='color:green;'>🌟 Odlična prilika!</b>"

                    dist_tekst = f"Walk Score: <b>{row['Walk_Score']}/100</b><br>" if pd.notna(row.get('Walk_Score')) else ""

                    html = f"""
                    <div style="width:250px;">
                        <h5>{row['Naslov']}</h5>
                        <p style="font-size:12px; color:gray;">{row.get('Ulica', '')}, {row['Deo_Grada']}</p>
                        <hr>
                        <b>Cena:</b> {row['Cena_EUR']:,.0f} €<br>
                        <b>Kvadratura:</b> {row['Kvadratura_m2']} m²<br>
                        <b>Cena/m²:</b> {row['Cena_po_m2']:,.0f} €<br>
                        {dist_tekst}
                        <small>AI Procena: ~{row['Predvidjena_Cena']:,.0f} €</small><br>
                        {status}<br>
                        <a href="{row['Link']}" target="_blank" style="display:block; margin-top:10px; background:#007BFF; color:white; text-align:center; padding:5px; border-radius:5px; text-decoration:none;">Pogledaj oglas</a>
                    </div>
                    """
                    folium.Marker(
                        location=[row['Latitude'], row['Longitude']],
                        popup=folium.Popup(html, max_width=300),
                        icon=folium.Icon(color=boja, icon=ikona, prefix="fa")
                    ).add_to(marker_cluster)

                st_folium(m, width="100%", height=500, returned_objects=[])
        else:
            st.warning("Nema stanova sa ispravnim GPS koordinatama za prikaz na mapi.")

        st.subheader("📋 Brzi pregled i dodavanje u omiljene")
        if len(f_df) > 0:
            df_prikaz = f_df[['Naslov', 'Cena_EUR', 'Kvadratura_m2', 'Cena_po_m2', 'Deo_Grada', 'Link']].copy()
            df_prikaz.insert(0, "Sačuvaj", False)

            edited_df = st.data_editor(df_prikaz, hide_index=True, use_container_width=True)
            sacuvani_sada = edited_df[edited_df["Sačuvaj"] == True]

            if st.button("Sačuvaj izabrane oglase u 'Omiljeno'"):
                if not sacuvani_sada.empty:
                    for _, row in sacuvani_sada.iterrows():
                        if row['Link'] not in [x['Link'] for x in st.session_state['omiljeni']]:
                            st.session_state['omiljeni'].append(row.to_dict())
                    st.success(f"Dodato {len(sacuvani_sada)} oglasa u Omiljene!")

    # ---------------- PRIKAZ 2: ANALITIKA & KREDITI ----------------
    elif prikaz == "📈 Analitika & Krediti":
        st.header("Kalkulator Kredita i ROI Analitika")
        st.subheader("📊 Tržišni Trend: Odnos Cene i Kvadrature")
        if len(f_df) > 0:
            st.scatter_chart(data=f_df, x='Kvadratura_m2', y='Cena_EUR', color='Dobra_Prilika', size='Cena_po_m2')

        st.markdown("---")
        col_roi, col_kredit = st.columns(2)

        with col_roi:
            st.subheader("📈 Povrat Investicije (ROI)")
            if rezim == "Kupovina stanova":
                pretpostavljena_kirija = st.slider("Pretpostavljena mesečna kirija po m²:", 5, 25, 10)
                ocekivana_mesecna = f_df['Kvadratura_m2'].mean() * pretpostavljena_kirija
                godisnja_zarada = ocekivana_mesecna * 12
                godina_za_povrat = f_df['Cena_EUR'].mean() / godisnja_zarada if godisnja_zarada > 0 else 0
                st.info(f"Očekivana mesečna kirija za prosečan stan: **~{ocekivana_mesecna:,.0f} €**")
                st.success(f"Vreme isplate investicije (ROI): **{godina_za_povrat:.1f} godina**")
            else:
                st.warning("Prebacite na Kupovinu stanova za ROI analitiku.")

        with col_kredit:
            st.subheader("🏦 Kalkulator stambenog kredita")
            if rezim == "Kupovina stanova" and len(f_df) > 0:
                cena_stana = f_df['Cena_EUR'].mean()
                st.write(f"Računamo na prosečnu cenu: **{cena_stana:,.0f} €**")
                ucesce_proc = st.slider("Učešće (%)", 10, 50, 20)
                kamata = st.slider("Kamatna stopa (%)", 1.0, 10.0, 4.5, 0.1)
                godine = st.slider("Rok otplate (godina)", 5, 30, 20)

                glavnica = cena_stana * (1 - ucesce_proc / 100)
                r = (kamata / 100) / 12
                n = godine * 12
                rata = glavnica * (r * (1 + r) ** n) / ((1 + r) ** n - 1) if r > 0 else glavnica / n

                st.metric("Potrebno učešće", f"{cena_stana * (ucesce_proc / 100):,.0f} €")
                st.error(f"**Mesečna rata: {rata:,.0f} €**")

    # ---------------- PRIKAZ 3: A/B POREĐENJE ----------------
    elif prikaz == "⚖️ A/B Poređenje Lokacija":
        st.header("Uporedi dve mikrolokacije")
        svi_delovi = sorted(df['Deo_Grada'].dropna().unique().tolist())
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lokacija A")
            deo_A = st.selectbox("Izaberi deo grada A:", svi_delovi, key="A")
            podaci_A = df[df['Deo_Grada'] == deo_A]
            if len(podaci_A) > 0:
                st.metric("Broj oglasa", len(podaci_A))
                st.metric(f"Prosečna {label_cena}", f"{podaci_A['Cena_EUR'].mean():,.0f} €")
                st.metric("Walk Score (Pešačka zona)", f"{podaci_A['Walk_Score'].mean():,.0f} / 100")
        with col2:
            st.subheader("Lokacija B")
            deo_B = st.selectbox("Izaberi deo grada B:", svi_delovi, key="B", index=1 if len(svi_delovi) > 1 else 0)
            podaci_B = df[df['Deo_Grada'] == deo_B]
            if len(podaci_B) > 0:
                st.metric("Broj oglasa", len(podaci_B))
                st.metric(f"Prosečna {label_cena}", f"{podaci_B['Cena_EUR'].mean():,.0f} €")
                st.metric("Walk Score", f"{podaci_B['Walk_Score'].mean():,.0f} / 100")

    # ---------------- PRIKAZ 4: OMILJENI OGLASI ----------------
    elif prikaz == "⭐ Omiljeni Oglasi":
        st.header("Tvoja lista sačuvanih nekretnina")
        if len(st.session_state['omiljeni']) == 0:
            st.info("Trenutno nemaš sačuvanih oglasa. Idi na mapu i dodaj ih!")
        else:
            omiljeni_df = pd.DataFrame(st.session_state['omiljeni'])
            st.dataframe(omiljeni_df[['Naslov', 'Cena_EUR', 'Deo_Grada', 'Link']], use_container_width=True)
            if st.button("🗑️ Obriši sve omiljene"):
                st.session_state['omiljeni'] = []
                st.rerun()

    # ---------------- PRIKAZ 5: INTEGRACIJE ----------------
    elif prikaz == "⚙️ Integracije (Webhook)":
        st.header("API & Hardver Integracije")
        webhook_url = st.text_input("Unesite Webhook URL:", "")
        if st.button("🚀 Pošalji probni signal"):
            if webhook_url:
                st.success(f"Signal uspešno 'poslat' na {webhook_url}")
            else:
                st.warning("Prvo unesite URL!")

else:
    st.warning("Trenutno nema podataka na serveru. Proverite vezu sa API-jem.")
