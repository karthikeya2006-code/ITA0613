"""
Dataset Generator for Climate-Resilient Crop Yield Forecasting
Generates >= 10,000 realistic agro-climatic records covering multi-year, multi-region observations.
"""

import os
import numpy as np
import pandas as pd

def generate_crop_dataset(num_records: int = 12500, random_state: int = 42, output_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic yet realistic district-level agro-climatic data for India.
    Includes climate, soil, agronomic inputs, and yield targets.
    """
    np.random.seed(random_state)
    
    states_districts = {
        'Punjab': ['Ludhiana', 'Amritsar', 'Patiala', 'Jalandhar', 'Bathinda'],
        'Haryana': ['Karnal', 'Hisar', 'Ambala', 'Rohtak', 'Sirsa'],
        'Uttar Pradesh': ['Varanasi', 'Lucknow', 'Meerut', 'Agra', 'Gorakhpur'],
        'Madhya Pradesh': ['Indore', 'Bhopal', 'Ujjain', 'Jabalpur', 'Gwalior'],
        'Maharashtra': ['Pune', 'Nashik', 'Nagpur', 'Aurangabad', 'Solapur'],
        'Tamil Nadu': ['Coimbatore', 'Thanjavur', 'Madurai', 'Salem', 'Tiruchirappalli'],
        'Andhra Pradesh': ['Guntur', 'Krishna', 'Kurnool', 'Visakhapatnam', 'Chittoor'],
        'Karnataka': ['Mandya', 'Belagavi', 'Mysuru', 'Dharwad', 'Ballari'],
        'West Bengal': ['Burdwan', 'Hooghly', 'Murshidabad', 'Nadia', 'Midnapore'],
        'Gujarat': ['Rajkot', 'Ahmedabad', 'Surat', 'Vadodara', 'Junagadh']
    }
    
    crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Pulses', 'Millets', 'Groundnut']
    seasons = ['Kharif', 'Rabi', 'Zaid', 'Whole Year']
    years = list(range(2010, 2025))  # 15 years: 2010-2024
    
    records = []
    
    # Base crop characteristics (base yield t/ha, optimal rain mm, optimal temp C, optimal pH)
    crop_profiles = {
        'Wheat': {'base_yield': 4.2, 'opt_rain': 450, 'opt_temp': 20.0, 'opt_ph': 6.8, 'season_pref': 'Rabi'},
        'Rice': {'base_yield': 3.8, 'opt_rain': 1200, 'opt_temp': 28.0, 'opt_ph': 6.5, 'season_pref': 'Kharif'},
        'Maize': {'base_yield': 3.2, 'opt_rain': 750, 'opt_temp': 26.0, 'opt_ph': 6.5, 'season_pref': 'Kharif'},
        'Cotton': {'base_yield': 1.8, 'opt_rain': 700, 'opt_temp': 29.0, 'opt_ph': 7.2, 'season_pref': 'Kharif'},
        'Sugarcane': {'base_yield': 75.0, 'opt_rain': 1500, 'opt_temp': 30.0, 'opt_ph': 7.0, 'season_pref': 'Whole Year'},
        'Pulses': {'base_yield': 1.1, 'opt_rain': 400, 'opt_temp': 24.0, 'opt_ph': 7.0, 'season_pref': 'Rabi'},
        'Millets': {'base_yield': 1.5, 'opt_rain': 350, 'opt_temp': 31.0, 'opt_ph': 7.2, 'season_pref': 'Kharif'},
        'Groundnut': {'base_yield': 2.0, 'opt_rain': 550, 'opt_temp': 27.0, 'opt_ph': 6.4, 'season_pref': 'Kharif'}
    }
    
    # State baselines (regional climate variation)
    state_climates = {
        'Punjab': {'rain_mean': 650, 'temp_mean': 24.0, 'humidity_mean': 58},
        'Haryana': {'rain_mean': 580, 'temp_mean': 25.0, 'humidity_mean': 55},
        'Uttar Pradesh': {'rain_mean': 950, 'temp_mean': 26.0, 'humidity_mean': 65},
        'Madhya Pradesh': {'rain_mean': 1050, 'temp_mean': 27.0, 'humidity_mean': 60},
        'Maharashtra': {'rain_mean': 1150, 'temp_mean': 27.5, 'humidity_mean': 68},
        'Tamil Nadu': {'rain_mean': 980, 'temp_mean': 29.0, 'humidity_mean': 72},
        'Andhra Pradesh': {'rain_mean': 920, 'temp_mean': 28.5, 'humidity_mean': 70},
        'Karnataka': {'rain_mean': 1100, 'temp_mean': 26.5, 'humidity_mean': 69},
        'West Bengal': {'rain_mean': 1550, 'temp_mean': 27.0, 'humidity_mean': 80},
        'Gujarat': {'rain_mean': 820, 'temp_mean': 28.0, 'humidity_mean': 62}
    }
    
    for i in range(num_records):
        state = np.random.choice(list(states_districts.keys()))
        district = np.random.choice(states_districts[state])
        crop = np.random.choice(crops)
        year = int(np.random.choice(years))
        
        # Season selection with slight bias toward preferred season
        pref_season = crop_profiles[crop]['season_pref']
        season = pref_season if np.random.rand() < 0.65 else np.random.choice(seasons)
        
        # Agro-climatic values
        base_clim = state_climates[state]
        annual_rainfall = max(100.0, np.random.normal(base_clim['rain_mean'], 180.0))
        avg_temp = np.random.normal(base_clim['temp_mean'], 3.2)
        humidity = np.clip(np.random.normal(base_clim['humidity_mean'], 8.5), 20.0, 98.0)
        
        # Soil parameters
        soil_n = np.clip(np.random.normal(240, 50), 60, 480)    # kg/ha
        soil_p = np.clip(np.random.normal(35, 12), 8, 85)       # kg/ha
        soil_k = np.clip(np.random.normal(190, 45), 50, 380)    # kg/ha
        soil_ph = np.clip(np.random.normal(6.8, 0.6), 4.5, 9.0)
        
        # Agronomic inputs
        area_ha = np.clip(np.random.exponential(1200) + 100, 50, 25000)
        fertilizer_kg = np.clip(np.random.normal(140, 35), 20, 320)
        pesticide_kg = np.clip(np.random.normal(2.2, 0.8), 0.1, 6.0)
        
        # Response function (Yield)
        prof = crop_profiles[crop]
        
        # Rainfall penalty
        rain_diff = (annual_rainfall - prof['opt_rain']) / prof['opt_rain']
        rain_factor = np.exp(-0.5 * (rain_diff ** 2))
        
        # Temperature penalty
        temp_diff = (avg_temp - prof['opt_temp']) / 6.0
        temp_factor = np.exp(-0.5 * (temp_diff ** 2))
        
        # Soil & nutrient factor
        ph_factor = np.exp(-1.5 * ((soil_ph - prof['opt_ph']) ** 2))
        nutrient_boost = (soil_n / 250.0 + soil_p / 35.0 + soil_k / 200.0) / 3.0
        fert_boost = 0.8 + 0.4 * (fertilizer_kg / 150.0)
        pest_protection = 0.9 + 0.2 * (pesticide_kg / 2.5)
        
        # Base calculation
        expected_yield = prof['base_yield'] * (0.35 * rain_factor + 0.35 * temp_factor + 0.30 * ph_factor) * nutrient_boost * fert_boost * pest_protection
        # Add realistic noise
        actual_yield = max(0.1, expected_yield + np.random.normal(0, 0.12 * prof['base_yield']))
        
        # Scale sugarcane yields down to standard ton/ha metric representation if needed or keep standard
        # Let's standardize crop yield scale: Sugarcane base ~ 70 t/ha, others 1-6 t/ha.
        # To facilitate regression comparisons across crops without extreme skew, we can normalize or keep as realistic t/ha.
        
        records.append({
            'Record_ID': f'REC_{year}_{i+1:06d}',
            'Year': year,
            'State': state,
            'District': district,
            'Crop': crop,
            'Season': season,
            'Area_Hectares': round(float(area_ha), 2),
            'Annual_Rainfall_mm': round(float(annual_rainfall), 2),
            'Avg_Temperature_C': round(float(avg_temp), 2),
            'Humidity_Percent': round(float(humidity), 2),
            'Soil_N_kg_per_ha': round(float(soil_n), 2),
            'Soil_P_kg_per_ha': round(float(soil_p), 2),
            'Soil_K_kg_per_ha': round(float(soil_k), 2),
            'Soil_pH': round(float(soil_ph), 2),
            'Fertilizer_Usage_kg_per_ha': round(float(fertilizer_kg), 2),
            'Pesticide_Usage_kg_per_ha': round(float(pesticide_kg), 2),
            'Yield_tonnes_per_ha': round(float(actual_yield), 3)
        })
        
    df = pd.DataFrame(records)
    
    # Introduce ~1.5% realistic missing values in continuous weather/soil to test data pipeline cleaning
    mask_rain = np.random.rand(len(df)) < 0.015
    df.loc[mask_rain, 'Annual_Rainfall_mm'] = np.nan
    mask_temp = np.random.rand(len(df)) < 0.012
    df.loc[mask_temp, 'Avg_Temperature_C'] = np.nan
    mask_ph = np.random.rand(len(df)) < 0.010
    df.loc[mask_ph, 'Soil_pH'] = np.nan
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} records saved to {output_path}")
        
    return df

if __name__ == '__main__':
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_yield_dataset.csv')
    generate_crop_dataset(num_records=12500, output_path=dataset_path)
