"""Access to the renewables.ninja API"""

import requests
import json
import time
import pandas as pd


def request_PV(
    token, lat, lon, date_from, date_to, capacity, system_loss, tracking, tilt, azim
):

    # requests are limited to 50/hour which works out to 1 per 72 seconds
    # add 2 seconds safety margin
    time.sleep(72 + 10)

    api_base = "https://www.renewables.ninja/api/"
    s = requests.session()
    # Send token header with each request
    s.headers = {"Authorization": "Token " + token}

    ##
    # PV example
    ##

    url = api_base + "data/pv"

    args = {
        "lat": lat,
        "lon": lon,
        "date_from": date_from,
        "date_to": date_to,
        "dataset": "merra2",
        "capacity": capacity,
        "system_loss": system_loss,
        "tracking": tracking,
        "tilt": tilt,
        "azim": azim,
        "format": "json",
    }

    r = s.get(url, params=args)

    # Parse JSON to get a pandas.DataFrame of data and dict of metadata
    parsed_response = json.loads(r.text)
    data = pd.read_json(json.dumps(parsed_response["data"]), orient="index")
    metadata = parsed_response["metadata"]

    return {"data": data, "metadata": metadata}


def request_wind(token, lat, lon, date_from, date_to, capacity, height, turbine):

    # requests are limited to 50/hour which works out to 1 per 72 seconds
    # add 2 seconds safety margin
    time.sleep(72 + 10)

    api_base = "https://www.renewables.ninja/api/"
    s = requests.session()
    # Send token header with each request
    s.headers = {"Authorization": "Token " + token}

    ##
    # Wind example
    ##

    url = api_base + "data/wind"

    args = {
        "lat": lat,
        "lon": lon,
        "date_from": date_from,
        "date_to": date_to,
        "capacity": capacity,
        "height": height,
        "turbine": turbine,
        "format": "json",
    }

    r = s.get(url, params=args)

    parsed_response = json.loads(r.text)
    data = pd.read_json(json.dumps(parsed_response["data"]), orient="index")
    metadata = parsed_response["metadata"]

    return {"data": data, "metadata": metadata}


if __name__ == "__main__":

    # insert own API token
    token = "INSERT_OWN_API_TOKEN"
    lat = 34.125
    lon = 39.814
    date_from = "2015-01-01"
    date_to = "2015-12-31"
    capacity = 1.0
    system_loss = 0.1
    tracking = 0
    tilt = 35
    azim = 180

    capacity_wind = 1.0
    height = 100
    turbine = "Vestas V80 2000"

    PV = request_PV(
        token, lat, lon, date_from, date_to, capacity, system_loss, tracking, tilt, azim
    )
    print(PV["data"])

    wind = request_wind(token, lat, lon, date_from, date_to, capacity, height, turbine)
    print(wind["data"])
