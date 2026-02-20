import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# --- KONFIGURACIJA ---
# Koristimo /20/ jer smo provalili da server zapravo vraća 50 oglasa!
BASE_URL = "https://www.nekretnine.rs/stambeni-objekti/stanovi/izdavanje-prodaja/prodaja/lista/po-stranici/20/stranica/"
BROJ_STRANICA = 418  # Kada budeš spreman, stavi ovde 100 ili 150

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "sr-RS,sr;q=0.9,en-US;q=0.8,en;q=0.7",
}


def izvuci_najvecu_cenu(tekst):
    """Izvlači pravu cenu stana i ignoriše cenu po kvadratu."""
    if not tekst: return None
    clean_text = tekst.replace('EUR', ' ').replace('€', ' ').replace('/m2', ' ').replace('.', '')

    if "EUR" in tekst:
        glavni_deo = tekst.split("EUR")[0]
    elif "€" in tekst:
        glavni_deo = tekst.split("€")[0]
    else:
        glavni_deo = tekst

    samo_brojevi = re.sub(r'[^\d]', '', glavni_deo)
    try:
        return int(samo_brojevi)
    except:
        return None


def izvuci_detaljnu_lokaciju(oglas_url):
    """Ulazi u oglas i vadi Grad, Deo grada i Ulicu."""
    time.sleep(random.uniform(1.2, 2.0))  # Obavezna pauza pre ulaska u oglas!
    grad, deo_grada, ulica = "N/A", "N/A", "N/A"

    try:
        res = requests.get(oglas_url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            lokacija_div = soup.find('div', class_='property__location')

            if lokacija_div:
                lista_lokacija = lokacija_div.find_all('li')
                delovi = [li.get_text(strip=True) for li in lista_lokacija]

                if len(delovi) > 2: grad = delovi[2]
                if len(delovi) > 3: deo_grada = delovi[3]
                if len(delovi) > 4: ulica = delovi[4]
    except Exception:
        pass  # Ignorišemo grešku ako sajt slučajno ne učita jedan oglas

    return grad, deo_grada, ulica


def parsiraj_stranicu(url):
    """Parsira glavnu stranicu i šalje svaki oglas na dodatnu obradu."""
    print(f"\nOtvaram glavnu stranicu: {url}")
    print("  -> Čekam odgovor servera...")
    time.sleep(random.uniform(1.5, 3))

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        print(f"  -> Odgovor primljen! Status: {response.status_code}")
    except Exception as e:
        print(f"Greška u konekciji: {e}")
        return []

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    oglasi = soup.find_all('div', class_='row offer')
    ukupno_na_stranici = len(oglasi)  # Dinamički brojimo oglase (biće ih 50)
    podaci_sa_stranice = []

    for idx, oglas in enumerate(oglasi, 1):
        try:
            # 1. Naslov i Link
            naslov_tag = oglas.find('h2', class_='offer-title')
            naslov, link = "N/A", "N/A"
            if naslov_tag:
                a_tag = naslov_tag.find('a')
                if a_tag:
                    naslov = a_tag.get_text(strip=True)
                    link = "https://www.nekretnine.rs" + a_tag['href']

            # 2. Lokacija osnovna
            lokacija = "N/A"
            lokacija_tag = oglas.find('p', class_='offer-location')
            if lokacija_tag:
                lokacija = lokacija_tag.get_text(strip=True)

            # 3. Cena
            cena = None
            cena_container = oglas.find('p', class_='offer-price') or oglas.find(class_='stickyBox__price')
            if cena_container:
                klase = cena_container.get('class', [])
                if 'offer-price--invert' not in klase:
                    raw_text = cena_container.get_text(strip=True)
                    cena = izvuci_najvecu_cenu(raw_text)

            # 4. Kvadratura
            kvadratura = None
            kvad_container = oglas.find('p', class_='offer-price--invert') or oglas.find(class_='stickyBox__size')
            if kvad_container:
                raw_text = kvad_container.get_text(strip=True)
                clean_kvad = re.sub(r'[^\d.]', '',
                                    raw_text.lower().replace('m²', '').replace('m2', '').replace(',', '.'))
                try:
                    kvadratura = float(clean_kvad)
                except:
                    pass

            # --- 5. ULAZAK U OGLAS ---
            # Skeniramo detalje samo ako stan ima validnu cenu (>10k) i kvadraturu
            if cena and kvadratura and cena > 10000:
                print(f"    Skeniram oglas {idx}/{ukupno_na_stranici} za detalje (Ulica)...", end='\r')
                grad, deo_grada, ulica = izvuci_detaljnu_lokaciju(link)

                podaci_sa_stranice.append({
                    'Naslov': naslov,
                    'Grad': grad,
                    'Deo_Grada': deo_grada,
                    'Ulica': ulica,
                    'Lokacija_Osnovna': lokacija,
                    'Cena_EUR': cena,
                    'Kvadratura_m2': kvadratura,
                    'Link': link
                })

        except Exception as e:
            continue

    print(f"  -> Uspešno obrađeno {len(podaci_sa_stranice)} stanova na ovoj stranici.    ")
    return podaci_sa_stranice


def main():
    print(f"--- POČETAK SKRAPINGA SA ULICAMA ---")
    svi_stanovi = []

    for i in range(1, BROJ_STRANICA + 1):
        url = f"{BASE_URL}{i}/"
        novi_podaci = parsiraj_stranicu(url)
        svi_stanovi.extend(novi_podaci)

        # --- BACKUP SISTEM ---
        # Posle svake stranice čuvamo ono što imamo, za svaki slučaj
        if svi_stanovi:
            backup_df = pd.DataFrame(svi_stanovi)
            backup_df.to_csv("nekretnine_backup.csv", index=False, encoding='utf-8-sig')

    if not svi_stanovi:
        print("\n[GREŠKA] Nema prikupljenih podataka.")
        return

    # --- FINALNA OBRADA ---
    df = pd.DataFrame(svi_stanovi)

    # Proračun cene po kvadratu
    df['Cena_po_m2'] = (df['Cena_EUR'] / df['Kvadratura_m2']).round(2)

    # Redosled kolona
    kolone = ['Grad', 'Deo_Grada', 'Ulica', 'Cena_EUR', 'Kvadratura_m2', 'Cena_po_m2', 'Naslov', 'Link',
              'Lokacija_Osnovna']
    df = df[kolone]

    file_name = "nekretnine_sa_ulicama.csv"
    df.to_csv(file_name, index=False, encoding='utf-8-sig')

    print(f"\nUspešno završeno! Sačuvano u: {file_name}")
    print("\n--- PROVERA REZULTATA (Prvih 5 redova) ---")
    print(df[['Grad', 'Deo_Grada', 'Ulica', 'Cena_EUR']].head())


if __name__ == "__main__":
    main()