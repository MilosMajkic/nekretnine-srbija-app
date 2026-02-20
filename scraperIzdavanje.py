import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

# --- KONFIGURACIJA ZA IZDAVANJE ---
BASE_URL = "https://www.nekretnine.rs/stambeni-objekti/stanovi/izdavanje-prodaja/izdavanje/lista/po-stranici/20/stranica/"
BROJ_STRANICA = 198 # Povećaj ovo kada budeš hteo da skineš sve

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "sr-RS,sr;q=0.9,en-US;q=0.8,en;q=0.7",
}


def izvuci_najvecu_cenu(tekst):
    """Izvlači pravu kiriju i ignoriše cenu po kvadratu."""
    if not tekst: return None

    # Prvo sečemo na "EUR" ili "€" da bismo odbacili onaj span tag sa slike
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
    """Ulazi u sam oglas i vadi Grad, Deo grada i Ulicu."""
    time.sleep(random.uniform(1.2, 2.0))  # Pauza da ne blokiraju IP
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
        pass

    return grad, deo_grada, ulica


def parsiraj_stranicu(url):
    print(f"\nOtvaram glavnu stranicu: {url}")
    print("  -> Čekam odgovor servera...")
    time.sleep(random.uniform(1.5, 3))

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
    except Exception as e:
        print(f"Greška u konekciji: {e}")
        return []

    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    oglasi = soup.find_all('div', class_='row offer')
    ukupno_na_stranici = len(oglasi)
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

            # 3. Kirija (Ovo gađa tačno tvoj stickyBox__price sa slike)
            cena = None
            cena_container = oglas.find('p', class_='offer-price') or oglas.find(class_='stickyBox__price')
            if cena_container:
                # Pre čitanja teksta, fizički brišemo span tag sa cenom po kvadratu
                for span in cena_container.find_all('span'):
                    span.decompose()

                raw_text = cena_container.get_text(strip=True)
                cena = izvuci_najvecu_cenu(raw_text)

            # 4. Kvadratura (Ovo gađa tačno tvoj stickyBox__size sa slike)
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
            # Za izdavanje uzimamo stanove čija je kirija preko 100 EUR
            if cena and kvadratura and cena > 100:
                print(f"    Skeniram oglas {idx}/{ukupno_na_stranici} za detalje (Ulica)...", end='\r')
                grad, deo_grada, ulica = izvuci_detaljnu_lokaciju(link)

                podaci_sa_stranice.append({
                    'Naslov': naslov,
                    'Grad': grad,
                    'Deo_Grada': deo_grada,
                    'Ulica': ulica,
                    'Lokacija_Osnovna': lokacija,
                    'Kirija_EUR': cena,
                    'Kvadratura_m2': kvadratura,
                    'Link': link
                })

        except Exception as e:
            continue

    print(f"  -> Uspešno obrađeno {len(podaci_sa_stranice)} stanova na ovoj stranici.    ")
    return podaci_sa_stranice


def main():
    print(f"--- POČETAK SKRAPINGA ZA IZDAVANJE ---")
    svi_stanovi = []

    for i in range(1, BROJ_STRANICA + 1):
        url = f"{BASE_URL}{i}/"
        novi_podaci = parsiraj_stranicu(url)
        svi_stanovi.extend(novi_podaci)

        # --- BACKUP ---
        if svi_stanovi:
            backup_df = pd.DataFrame(svi_stanovi)
            backup_df.to_csv("stanovi_izdavanje_backup.csv", index=False, encoding='utf-8-sig')

    if not svi_stanovi:
        print("\n[GREŠKA] Nema prikupljenih podataka.")
        return

    # --- FINALNA OBRADA ---
    df = pd.DataFrame(svi_stanovi)

    # Računamo koliko košta kvadrat mesečno (npr. 10 eur/m2)
    df['Kirija_po_m2'] = (df['Kirija_EUR'] / df['Kvadratura_m2']).round(2)

    # Ređamo kolone
    kolone = ['Grad', 'Deo_Grada', 'Ulica', 'Kirija_EUR', 'Kvadratura_m2', 'Kirija_po_m2', 'Naslov', 'Link',
              'Lokacija_Osnovna']
    df = df[kolone]

    file_name = "stanovi_izdavanje_final.csv"
    df.to_csv(file_name, index=False, encoding='utf-8-sig')

    print(f"\nUspešno završeno! Sačuvano u: {file_name}")
    print("\n--- PROVERA REZULTATA (Prvih 5 redova) ---")
    pd.set_option('display.max_columns', None)
    print(df[['Grad', 'Deo_Grada', 'Ulica', 'Kirija_EUR', 'Kvadratura_m2']].head())


if __name__ == "__main__":
    main()