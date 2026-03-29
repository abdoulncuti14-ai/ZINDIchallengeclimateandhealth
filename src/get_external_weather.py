import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os

# ✅ CORRECTION DE L'ERREUR : Utilisation de CachedSession
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def get_external_data():
    print("🌍 Connexion à l'API Open-Meteo...")
    # On charge les coordonnées uniques pour ne pas ralentir l'API
    train = pd.read_csv('data/raw/Train.csv')
    test = pd.read_csv('data/raw/Test.csv')
    coords = pd.concat([train, test])[['latitude', 'longitude']].drop_duplicates()

    results = []
    for _, row in coords.iterrows():
        try:
            params = {
                "latitude": row['latitude'],
                "longitude": row['longitude'],
                "start_date": "2021-01-01",
                "end_date": "2024-12-31",
                "daily": ["relative_humidity_2m_max", "surface_pressure_mean"],
                "timezone": "auto"
            }
            responses = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
            res = responses[0].Daily()

            results.append({
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'ext_humidity': res.Variables(0).ValuesAsNumpy().mean(),
                'ext_pressure': res.Variables(1).ValuesAsNumpy().mean()
            })
        except Exception as e:
            continue

    df_ext = pd.DataFrame(results)
    os.makedirs('data/processed', exist_ok=True)
    df_ext.to_csv('data/processed/external_data.csv', index=False)
    print("✅ Fichier data/processed/external_data.csv créé !")


if __name__ == "__main__":
    get_external_data()