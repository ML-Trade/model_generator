"""
Check data in local directory -> is it sufficient?
If not, check to see if required data is in Google Drive. Get what you can.
Need more data? Use polygon.io API

Once you have all data, save it locally
"""

"""
Filename convention:

data folder:
For < hour
{symbol}-{minute}-{measurement}-{start}-{end}.csv
"""

import json
from os import environ, path
import requests
from datetime import timedelta, datetime


def get_time_delta(multiplier: int, measurement: str) -> timedelta:
    if measurement == "second":
        return timedelta(seconds = multiplier)
    elif measurement == "minute":
        return timedelta(minutes = multiplier)
    elif measurement == "hour":
        return timedelta(hours = multiplier)
    elif measurement == "day":
        return timedelta(days = multiplier)
    elif measurement == "week":
        return timedelta(weeks = multiplier)
    elif measurement == "month":
        return timedelta(days = 30.4167 * multiplier)
    elif measurement == "year":
        return timedelta(days = 365 * multiplier)
    return timedelta(minutes = 1)

class DataUpdater:

    def __init__(self) -> None:
        token_file = open(path.join(environ["workspace"], "tokens.json"))
        tokens = json.load(token_file)
        token_file.close()
        self.github_token = tokens["github"]
        self.google_drive_token = tokens["google_drive"]
        self.polygonio_token = tokens["polygonio"]

        print(self.get_from_api("C:EURUSD", datetime(2021, 12, 28), datetime(2021, 12, 29)))

    def get_required_data(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "minute"):
        # Does data already exist?
        data_folder = path.join(environ["workspace"], "data")
        time_delta = get_time_delta(multiplier, measurement)

        if time_delta < timedelta(hours = 1):
            # Look for monthly csv files
            pass
        elif time_delta < timedelta(days = 1):
            # Look for yearly csv files
            pass
        else:
            # Look for decadely csv files
            pass


        
    
    def get_from_api(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "minute") -> dict:
        res: dict = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{measurement}/{start.date()}/{end.date()}",
            headers={ "Authorization": f"Bearer {self.polygonio_token}" }
        ).json()
        return res

