"""
Predictive Flash-Flood Model for a 4-Acre Malaysian Recreational Lake
TFB3033 Group Project — Intelligent Environmental Monitoring

This module handles:
1. Hydrological synthetic dataset generation tailored to tropical Malaysian climates (heavy monsoon downpours, high runoff).
2. Machine learning model training (Random Forest Regressor & Moving Average baseline) to forecast lake water levels.
3. Real-time inference, risk assessment (Normal/Alert/Critical), and time-to-flood estimation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

# Physical and hydrological constraints for a 4-acre (approx 16,187 m²) Malaysian lake
LAKE_AREA_M2 = 16187.0
NORMAL_WATER_LEVEL = 2.00     # meters (baseline depth)
ALERT_WATER_LEVEL = 3.00      # meters (spillway warning)
CRITICAL_WATER_LEVEL = 3.50   # meters (overtopping / flash flood stage)
MAX_DRAINAGE_CAPACITY = 15.0  # mm/h equivalent water level discharge rate

def generate_synthetic_lake_data(num_samples=2500, random_seed=42):
    """
    Generates realistic time-series hydrology data reflecting Malaysian tropical weather patterns:
    - Afternoon convective storms and torrential monsoon rains (up to 120 mm/h).
    - Lake level dynamics based on catchment inflow, lake storage, and culvert discharge.
    """
    np.random.seed(random_seed)
    
    # Rainfall intensity (mm/h): mixture of dry periods, moderate showers, and heavy tropical downpours
    rain_type = np.random.choice(['dry', 'moderate', 'heavy_monsoon'], size=num_samples, p=[0.55, 0.30, 0.15])
    rainfall_intensity = np.zeros(num_samples)
    
    for i, r_type in enumerate(rain_type):
        if r_type == 'dry':
            rainfall_intensity[i] = np.random.exponential(scale=1.5) if np.random.rand() > 0.4 else 0.0
        elif r_type == 'moderate':
            rainfall_intensity[i] = np.random.uniform(5.0, 30.0)
        else: # heavy monsoon cloudburst
            rainfall_intensity[i] = np.random.uniform(35.0, 110.0)
            
    rainfall_intensity = np.clip(rainfall_intensity, 0, 130.0)
    
    # 30-min moving average / cumulative rain effect
    rolling_rain_30m = pd.Series(rainfall_intensity).rolling(window=3, min_periods=1).mean().values
    
    # Initial lake levels (meters) with natural fluctuation around 2.0m
    current_water_level = np.random.uniform(1.70, 3.40, size=num_samples)
    
    # Rate of change of water level in the previous 15 minutes (m/15min)
    prev_rate_of_change = (rainfall_intensity * 0.005) - np.random.uniform(0.01, 0.04, size=num_samples)
    prev_rate_of_change = np.clip(prev_rate_of_change, -0.05, 0.30)
    
    # Soil saturation index (0.0 = completely dry, 1.0 = fully saturated monsoon ground)
    soil_saturation = np.random.beta(a=2, b=1.5, size=num_samples)
    
    # Runoff factor: higher soil saturation leads to faster runoff directly into the 4-acre lake
    effective_runoff = rainfall_intensity * (0.35 + 0.55 * soil_saturation)
    
    # Inflow surge into lake (meters of water level addition in 30 minutes)
    net_inflow_surge = (effective_runoff * 0.0075) + (rolling_rain_30m * 0.0035)
    
    # Culvert / spillway drainage capacity reduces water level
    drainage_discharge = np.clip(0.04 + (current_water_level - 1.80) * 0.03, 0.02, 0.12)
    
    # Forecasted water level in 30-45 minutes
    noise = np.random.normal(0, 0.02, size=num_samples)
    forecasted_water_level = current_water_level + net_inflow_surge - drainage_discharge + noise
    forecasted_water_level = np.clip(forecasted_water_level, 1.50, 4.20)
    
    df = pd.DataFrame({
        'current_water_level': np.round(current_water_level, 3),
        'rainfall_intensity': np.round(rainfall_intensity, 2),
        'rolling_rain_30m': np.round(rolling_rain_30m, 2),
        'water_level_roc': np.round(prev_rate_of_change, 3),
        'soil_saturation': np.round(soil_saturation, 2),
        'forecasted_water_level': np.round(forecasted_water_level, 3)
    })
    
    return df

class FloodPredictorModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.baseline_model = Ridge()
        self.features = ['current_water_level', 'rainfall_intensity', 'rolling_rain_30m', 'water_level_roc', 'soil_saturation']
        self.metrics = {}
        self.is_trained = False
        
    def train(self, df=None):
        if df is None:
            df = generate_synthetic_lake_data()
            
        X = df[self.features]
        y = df['forecasted_water_level']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Regressor
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Baseline model for comparison
        self.baseline_model.fit(X_train, y_train)
        y_pred_baseline = self.baseline_model.predict(X_test)
        
        # Calculate evaluation metrics for reporting
        self.metrics = {
            'r2_score': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'baseline_r2': float(r2_score(y_test, y_pred_baseline)),
            'baseline_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_baseline))),
            'feature_importances': dict(zip(self.features, self.model.feature_importances_))
        }
        self.is_trained = True
        return self.metrics

    def predict(self, current_level, rainfall_intensity, rolling_rain_30m=None, water_level_roc=0.02, soil_saturation=0.65):
        """
        Predicts future lake water level and computes risk tier.
        """
        if not self.is_trained:
            self.train()
            
        if rolling_rain_30m is None:
            rolling_rain_30m = rainfall_intensity * 0.85
            
        input_data = pd.DataFrame([{
            'current_water_level': current_level,
            'rainfall_intensity': rainfall_intensity,
            'rolling_rain_30m': rolling_rain_30m,
            'water_level_roc': water_level_roc,
            'soil_saturation': soil_saturation
        }])
        
        predicted_level = float(self.model.predict(input_data)[0])
        delta = predicted_level - current_level
        
        # Risk assessment categorization
        if predicted_level >= CRITICAL_WATER_LEVEL:
            risk_status = "CRITICAL"
            color = "#EF4444" # Red
            message = "FLASH FLOOD WARNING: Water level exceeds safety embankment (3.50m). Immediate spillway gate activation required!"
        elif predicted_level >= ALERT_WATER_LEVEL:
            risk_status = "ALERT"
            color = "#F59E0B" # Yellow/Amber
            message = "ELEVATED RISK: Water approaching critical retention limit (3.00m). Notify maintenance & prepare containment."
        else:
            risk_status = "NORMAL"
            color = "#10B981" # Green
            message = "SAFE LEVEL: Water level within normal catchment capacity (Below 3.00m)."
            
        # Estimate time to critical overtopping if water is rising
        if delta > 0 and predicted_level < CRITICAL_WATER_LEVEL:
            time_to_flood_min = int(round(((CRITICAL_WATER_LEVEL - current_level) / max(delta, 0.01)) * 30))
        elif predicted_level >= CRITICAL_WATER_LEVEL:
            time_to_flood_min = 0
        else:
            time_to_flood_min = -1 # Safe / receding
            
        return {
            'predicted_level': round(predicted_level, 2),
            'delta_meters': round(delta, 2),
            'risk_status': risk_status,
            'color': color,
            'message': message,
            'time_to_flood_min': time_to_flood_min
        }

if __name__ == '__main__':
    print("Training predictive flash-flood model for 4-acre lake...")
    predictor = FloodPredictorModel()
    results = predictor.train()
    print("Model Training Complete!")
    print(f"R² Score: {results['r2_score']:.4f}")
    print(f"RMSE: {results['rmse']:.4f} meters")
    print(f"MAE: {results['mae']:.4f} meters")
    print("\nFeature Importances:")
    for feat, imp in results['feature_importances'].items():
        print(f" - {feat}: {imp*100:.1f}%")
        
    # Sample prediction test
    test_pred = predictor.predict(current_level=2.85, rainfall_intensity=80.0, soil_saturation=0.85)
    print("\nSample Scenario: Monsoon Downpour (80 mm/h) with lake at 2.85m:")
    print(f"Predicted Level: {test_pred['predicted_level']}m | Risk: {test_pred['risk_status']} | Time to Flood: {test_pred['time_to_flood_min']} min")
