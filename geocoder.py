import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- KONFIGURACIJA ---
ULAZNI_FAJL = "stanovi_izdavanje_final.csv"
IZLAZNI_FAJL = "izdavanje_srb_geo.csv"


def main():
    print(f"--- Učitavam podatke iz {ULAZNI_FAJL} ---")
    try:
        df = pd.read_csv(ULAZNI_FAJL)
    except FileNotFoundError:
        print(f"GREŠKA: Nije pronađen fajl '{ULAZNI_FAJL}'.")
        return

    print(f"Učitano {len(df)} redova. Spremam se za geokodiranje cele Srbije...")

    # Inicijalizacija geolokatora
    geolocator = Nominatim(user_agent="srbija_nekretnine_skreper_v3")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.2)  # Malo veća pauza za svaki slučaj

    locations = []
    total = len(df)

    for index, row in df.iterrows():
        # Izvlačimo podatke i čistimo ih od 'N/A'
        grad = row['Grad'] if pd.notna(row['Grad']) and row['Grad'] != "N/A" else ""
        deo_grada = row['Deo_Grada'] if pd.notna(row['Deo_Grada']) and row['Deo_Grada'] != "N/A" else ""
        ulica = row['Ulica'] if pd.notna(row['Ulica']) and row['Ulica'] != "N/A" else ""

        # Pravimo listu pokušaja od najpreciznijeg do najopštijeg
        pokusaji = []

        # 1. Najpreciznije: Ulica, Deo Grada, Grad, Srbija
        if ulica and grad:
            if deo_grada:
                pokusaji.append(f"{ulica}, {deo_grada}, {grad}, Srbija")
            else:
                pokusaji.append(f"{ulica}, {grad}, Srbija")

        # 2. Srednje: Deo Grada, Grad, Srbija
        if deo_grada and grad:
            pokusaji.append(f"{deo_grada}, {grad}, Srbija")

        # 3. Opšte: Samo Grad, Srbija
        if grad:
            pokusaji.append(f"{grad}, Srbija")

        # Ako iz nekog razloga nemamo ni grad, preskačemo
        if not pokusaji:
            print(f"[{index + 1}/{total}] Preskačem: Nema podataka o lokaciji.")
            locations.append((None, None))
            continue

        # --- SISTEM REZERVNIH OPCIJA (FALLBACK) ---
        pronadjeno = False
        for adresa_za_pretragu in pokusaji:
            try:
                loc = geocode(adresa_za_pretragu)
                if loc:
                    locations.append((loc.latitude, loc.longitude))
                    print(
                        f"[{index + 1}/{total}] Nađeno: {adresa_za_pretragu} -> {loc.latitude:.4f}, {loc.longitude:.4f}")
                    pronadjeno = True
                    break  # Čim nađe, prekidamo petlju pokušaja
            except Exception as e:
                # Ako server prijavi grešku (timeout), ignorišemo i idemo dalje
                pass

        # Ako smo iscrpeli sve pokušaje, a nismo našli ništa
        if not pronadjeno:
            print(f"[{index + 1}/{total}] NIJE NAĐENO: {pokusaji[0]} (i sve rezervne opcije)")
            locations.append((None, None))

    # Dodavanje koordinata u DataFrame
    df['Latitude'], df['Longitude'] = zip(*locations)

    # Statistika
    broj_uspesnih = df['Latitude'].notna().sum()
    print(f"\n--- ZAVRŠENO ---")
    print(f"Uspešno locirano: {broj_uspesnih} od {total} stanova.")

    df.to_csv(IZLAZNI_FAJL, index=False, encoding='utf-8-sig')
    print(f"Geokodirani podaci sačuvani u '{IZLAZNI_FAJL}'.")


if __name__ == "__main__":
    main()