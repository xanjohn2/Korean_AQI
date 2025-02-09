from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import logging

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Load the data
df = pd.read_csv('df_final.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
logging.debug(f"Data shape: {df.shape}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_map_data')
def get_map_data():
    latest_data = df.sort_values('Date').groupby('Station code').last().reset_index()
    
    map_data = []
    for _, row in latest_data.iterrows():
        map_data.append({
            'station_code': row['Station code'],
            'district': row['District'],
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'aqi': row['AQI'],
            'category': row['AQI Category']
        })
    
    return jsonify(map_data)

@app.route('/get_forecast/<int:station_code>')
def get_forecast(station_code):
    station_data = df[df['Station code'] == station_code].sort_values('Date')
    
    if len(station_data) == 0:
        return jsonify({'error': 'Station not found'})
    
    # Select the AQI column for forecasting
    aqi_data = station_data['AQI']

    # Fit ARIMA model
    model = ARIMA(aqi_data, order=(5, 1, 0))  # (p, d, q) parameters
    model_fit = model.fit()

    # Forecast next 7 days
    forecast = model_fit.get_forecast(steps=7)
    forecast_dates = pd.date_range(start=aqi_data.index[-1], periods=8, freq='D')[1:]
    forecast_values = forecast.predicted_mean.tolist()
    conf_int = forecast.conf_int()

    return jsonify({
        'dates': forecast_dates.strftime('%Y-%m-%d').tolist(),
        'values': forecast_values,
        'conf_int_lower': conf_int.iloc[:, 0].tolist(),
        'conf_int_upper': conf_int.iloc[:, 1].tolist()
    })

@app.route('/get_historical/<int:station_code>')
def get_historical(station_code):
    station_data = df[df['Station code'] == station_code].sort_values('Date')
    
    if len(station_data) == 0:
        return jsonify({'error': 'Station not found'})
    
    recent_data = station_data.tail(30)
    
    return jsonify({
        'dates': recent_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'aqi': recent_data['AQI'].tolist()
    })

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    data = request.get_json()
    
    current_features = np.array([
        data['so2'], data['no2'], data['o3'],
        data['co'], data['pm10'], data['pm25']
    ]).reshape(1, -1)
    
    prediction = model.predict(current_features)[0]
    category = label_mapping.get(prediction, "Unknown")
    
    return jsonify({'aqi': float(prediction), 'category': category})

if __name__ == '__main__':
    app.run(debug=True)
