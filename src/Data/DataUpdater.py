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
from datetime import date, timedelta, datetime
from utils.polygon_api import format_symbol_for_api
import calendar
import pandas as pd


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
    

def get_last_day_of_month(date: datetime):
    (_, last_day_of_month) = calendar.monthrange(date.year, date.month)
    return last_day_of_month


class DataUpdater:

    def __init__(self) -> None:
        token_file = open(path.join(environ["workspace"], "tokens.json"))
        tokens = json.load(token_file)
        token_file.close()
        self.github_token = tokens["github"]
        self.google_drive_token = tokens["google_drive"]
        self.polygonio_token = tokens["polygonio"]

        # print(self.get_from_api("C:EURUSD", datetime(2021, 12, 28), datetime(2021, 12, 29)))
        self.get_required_data("EURUSD", datetime(2020, 10, 28), datetime(2021, 12, 29))

    def get_required_data(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "hour"):
        # Does data already exist?
        data_folder = path.join(environ["workspace"], "data")
        time_delta = get_time_delta(multiplier, measurement)

        def get_data_with_file_interval(file_interval: str):
            folder = path.join(data_folder, file_interval)
            
            adjusted_start = start if file_interval == "monthly" else start.replace(month = 1)
            adjusted_start = adjusted_start.replace(day = 1)
            adjusted_end = end if file_interval == "monthly" else end.replace(month = 12)
            adjusted_end = adjusted_end.replace(day = get_last_day_of_month(end))
            
            # Process one month at a time
            range_start = adjusted_start
            range_end = None
            while range_end != adjusted_end:
                range_end = range_start if file_interval == "monthly" else range_start.replace(month = 12)
                range_end = range_end.replace(day = get_last_day_of_month(range_end))
                file_path = path.join(folder, f"{symbol}-{multiplier}-{measurement}-{range_start.date()}-to-{range_end.date()}.csv")
                try:
                    df = pd.read_csv(file_path)
                    print(df)
                    print("Read from csv file")
                except:
                    # File did not exist; use polygon.io API then save the file
                    res = self.get_from_api(symbol, range_start, range_end, multiplier, measurement)
                    df = pd.DataFrame.from_dict(res["results"])
                    print(df)
                    print("Obtained from polygon.io")
                    df.to_csv(file_path, index = False)
                    # TODO: Sleep here to avoid hitting free resource tier limits (or pay $49 per month)
                    
                # 35 days is enough to always set to next month, 366 enough to always set to next year
                delta_days = 35 if file_interval == "monthly" else 366
                range_start = (range_start + timedelta(days = delta_days)).replace(day = 1)

        if time_delta < timedelta(hours = 1):
            get_data_with_file_interval("monthly")
        else:
            get_data_with_file_interval("yearly")



        
    
    def get_from_api(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "minute") -> dict:
        res: dict = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{format_symbol_for_api(symbol)}/range/{multiplier}/{measurement}/{start.date()}/{end.date()}?adjusted=true&sort=asc&limit=50000",
            headers={ "Authorization": f"Bearer {self.polygonio_token}" }
        ).json()
        return res

