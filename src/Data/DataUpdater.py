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
    


class DataUpdater:

    def __init__(self) -> None:
        token_file = open(path.join(environ["workspace"], "tokens.json"))
        tokens = json.load(token_file)
        token_file.close()
        self.github_token = tokens["github"]
        self.google_drive_token = tokens["google_drive"]
        self.polygonio_token = tokens["polygonio"]

        # print(self.get_from_api("C:EURUSD", datetime(2021, 12, 28), datetime(2021, 12, 29)))
        self.get_required_data("EURUSD", datetime(2021, 10, 28), datetime(2023, 12, 29))

    def get_required_data(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "minute"):
        # Does data already exist?
        data_folder = path.join(environ["workspace"], "data")
        time_delta = get_time_delta(multiplier, measurement)

        if time_delta < timedelta(hours = 1):
            folder = path.join(data_folder, "monthly")

            (_, last_day_of_month) = calendar.monthrange(end.year, end.month)
            adjusted_start = start.replace(day = 1)
            adjusted_end = end.replace(day = last_day_of_month)
            
            # Process one month at a time
            month_start = adjusted_start
            month_end = None
            while month_end != adjusted_end:
                (_, last_day_of_month) = calendar.monthrange(month_start.year, month_start.month)
                month_end = month_start.replace(day = last_day_of_month)
                file_path = path.join(folder, f"{symbol}-{multiplier}-{measurement}-{month_start.date()}-to-{month_end.date()}.csv")
                try:
                    df = pd.read_csv(file_path)
                    print(df)
                    print("Read from csv file")
                except:
                    # File did not exist
                    # Use polygon API then save the file
                    res = self.get_from_api(symbol, month_start, month_end, multiplier, measurement)
                    df = pd.DataFrame.from_dict(res["results"])
                    print(df)
                    print("Obtained from polygon.io")
                    df.to_csv(file_path, index = False)
                    exit()
                # 35 days is enough to always set to next month, then make it start of the month.
                month_start = (month_start + timedelta(days = 35)).replace(day = 1)
                

            # Look for monthly csv files
            # print(path.join(folder, f"{symbol}-{multiplier}-{measurement}-{start.date()}-to-{end.date()}.csv"))
        elif time_delta < timedelta(days = 1):
            # Look for yearly csv files
            pass
        else:
            # Look for decadely csv files
            pass


        
    
    def get_from_api(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "minute") -> dict:
        res: dict = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{format_symbol_for_api(symbol)}/range/{multiplier}/{measurement}/{start.date()}/{end.date()}?adjusted=true&sort=asc&limit=50000",
            headers={ "Authorization": f"Bearer {self.polygonio_token}" }
        ).json()
        return res

