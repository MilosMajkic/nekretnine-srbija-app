import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- KONFIGURACIJA ---
ULAZNI_FAJL = "izdavanje_srb_geo.csv"


def main():
    # 1. Učitavanje podataka
    print("--- 1. Učitavam podatke za ML ---")
    try:
        df = pd.read_csv(ULAZNI_FAJL)
    except FileNotFoundError:
        print(f"GREŠKA: Nema fajla '{ULAZNI_FAJL}'!")
        print("MORAŠ PRVO POKRENUTI 'geocoder.py' DA DOBIJEŠ OVAJ FAJL.")
        return

    # Provera da li imamo koordinate
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        print("GREŠKA: U fajlu ne postoje kolone Latitude/Longitude.")
        print("Pokreni geocoder.py ponovo.")
        return

    # 2. Čišćenje podataka
    initial_len = len(df)
    # Brišemo redove gde nema cene, kvadrature ILI koordinata
    df = df.dropna(subset=['Latitude', 'Longitude', 'Cena_EUR', 'Kvadratura_m2'])

    print(f"Od {initial_len} stanova, {len(df)} ima ispravne koordinate i cenu.")

    if len(df) < 10:
        print("\n[STOP] Imaš previše malo podataka (manje od 10) za treniranje modela.")
        print("Proveri 'geocoder.py' - možda nije uspeo da nađe adrese na mapi.")
        return

    # 3. Priprema za ML
    print("\n--- 2. Pripremam podatke ---")
    X = df[['Kvadratura_m2', 'Latitude', 'Longitude', 'Deo_Grada']]
    # Pretvaramo tekst (Deo Grada) u brojeve
    X = pd.get_dummies(X, columns=['Deo_Grada'], drop_first=True)
    y = df['Cena_EUR']

    # 4. Podela na Trening i Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Treniram model na {len(X_train)} stanova...")

    # 5. Kreiranje i treniranje modela
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Model je istreniran! Testiram...")

    # 6. Evaluacija
    predikcije = model.predict(X_test)
    mae = mean_absolute_error(y_test, predikcije)
    r2 = r2_score(y_test, predikcije)

    print(f"\n--- REZULTATI ---")
    print(f"Prosečna greška (MAE): {mae:,.0f} EUR")
    print(f"Preciznost (R2 Score): {r2:.2f}")

    # 7. Najvažniji faktori
    print("\n--- ŠTA NAJVIŠE UTIČE NA CENU? ---")
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    print(feature_importances.nlargest(5))


if __name__ == "__main__":
    main()