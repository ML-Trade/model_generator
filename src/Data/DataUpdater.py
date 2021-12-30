"""
Check data in local directory -> is it sufficient?
If not, check to see if required data is in Google Drive. Get what you can.
Need more data? Use polygon.io API

Once you have all data, save it locally
"""

import json
from os import environ, path
import requests
from datetime import datetime

class DataUpdater:

    def __init__(self) -> None:
        token_file = open(path.join(environ["workspace"], "tokens.json"))
        tokens = json.load(token_file)
        token_file.close()
        self.github_token = tokens["github"]
        self.google_drive_token = tokens["google_drive"]
        self.polygonio_token = tokens["polygonio"]

        self.get_from_api("C:EURUSD", datetime(2021, 12, 28), datetime(2021, 12, 29))
    
    def get_from_api(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "minute"):
        res: dict = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{measurement}/{start.date()}/{end.date()}",
            headers={ "Authorization": f"Bearer {self.polygonio_token}" }
        ).json()
        print(res)

