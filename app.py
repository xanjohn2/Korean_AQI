from flask import Flask, render_template, jsonify, request
import pandas as pd
import pickle
import numpy as np
from datetime import datetime, timedelta
import logging

app = Flask(__name__)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Load the data
df = pd.read_csv('df_final.csv')
df['Date'] = pd.to_datetime(df['Date'])
logging.debug(f"Data shape: {df.shape}")

# Load the trained model
with open('rfc.pkl', 'rb') as file:
    model = pickle.load(file)

# Label mapping
label_mapping = {
    0: 'Good',
    1: 'Moderate',
    2: 'Unhealthy',
    3: 'Very Unhealthy'
}

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
    
    latest_data = station_data.iloc[-1]
    
    forecast_dates = [(datetime.now() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(1, 7)]
    forecast_values = []
    
    current_features = np.array([
        latest_data['SO2'], latest_data['NO2'], latest_data['O3'],
        latest_data['CO'], latest_data['PM10'], latest_data['PM2.5']
    ]).reshape(1, -1)
    
    for _ in range(6):
        prediction = model.predict(current_features)[0]
        forecast_values.append(float(prediction))
        current_features = np.roll(current_features, -1)
        current_features[0][-1] = prediction
    
    return jsonify({
        'dates': forecast_dates,
        'values': forecast_values
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
    app.run()
