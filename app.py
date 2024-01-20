from flask import Flask , request
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import joblib
import numpy as np

# get all the farm regions
regions = {
    1: {
        "Name":"Region 1",
        "lat": 51,
        "long": 7.9,
        "N":20,
        "P":10,
        "K":20,
        "ph":7
    },
    2: {
        "Name": "Region 2",
        "lat": 20,
        "long":31.14,
        "N":20,
        "P":10,
        "K":20,
        "ph":7
    },
    3: {
        "Name": "Region 3",
        "lat": 70,
        "long":32.14,
        "N":20,
        "P":10,
        "K":20,
        "ph":7
    }
}

url = "https://api.open-meteo.com/v1/forecast"

app = Flask(__name__)

@app.route("/")
def getRegion():
        return regions , 200

def getWeatherData(id):
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    params = {
        "latitude": regions[id]['lat'],
        "longitude": regions[id]['long'],
        "hourly": ["temperature_2m","relative_humidity_2m","rain"],
        "timezone": "auto",
        "past_days": 7
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature"] = hourly_temperature_2m
    hourly_data["humidity"] = hourly_relative_humidity_2m
    hourly_data["rainfall"] = hourly_rain

    hourly_dataframe = pd.DataFrame(data = hourly_data)

    data = {
          "Latitude":float(response.Latitude()),
          "Longitude":float(response.Longitude()),
          "Timezone":str(response.Timezone()),
          "TimezoneAbbreviation":str(response.TimezoneAbbreviation()),
          "UtcOffset":str(response.UtcOffsetSeconds()),
          "weather": hourly_dataframe
    }

    return data 


model = joblib.load('recommender')

def predict(data):
      labels = ['apple','banana','blackgram','chickpea','coconut','coffee',
          'cotton','grapes','jute','kidneybeans','lentil','maize','mango',
          'mothbeans','mungbean','muskmelon','orange','papaya',
          'pigeonpeas','pomegranate','rice','watermelon']
      a, counts = np.unique(model.predict(data), return_counts=True)
      a = [labels[x] for x in np.unique(a)]
      percents = np.round(counts / sum(counts),2)
      pred = {k:v for (k,v) in zip(a,percents) if v > 0.1}
      return pred

@app.route("/data/<int:id>")
def getData(id):
    data = getWeatherData(id)
    parameters = data["weather"].drop(columns=['date'])
    parameters["N"] = regions[id]["N"]
    parameters["P"] = regions[id]["P"]
    parameters["K"] = regions[id]["K"]
    parameters["ph"] = regions[id]["ph"]
    data['Predictions'] = predict(parameters[["N","P", "K","temperature","humidity", "ph", "rainfall"]])

    data["weather"] = data["weather"].to_dict('records')
    return data

if __name__ == "__main__":
      app.run(debug=True)