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

# TODO: Remove this when / if you upgrade to the paid tier on polygon.io
IS_FREE_TIER = True

from enum import Enum
import json
import os
from typing import Any
import requests
from datetime import date, timedelta, datetime
from utils.polygon_api import format_symbol_for_api
import calendar
import pandas as pd
import time
from googleapiclient.discovery import build, mimetypes
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaFileUpload

FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"

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
        token_file = open(os.path.join(os.environ["workspace"], "tokens.json"))
        tokens: dict = json.load(token_file)
        if "PUT TOKEN HERE" in json.dumps(tokens):
            raise Exception("Tokens file not updated. Please add access tokens to tokens.json")
        token_file.close()
        self.github_token = tokens["github"]
        self.google_drive_token = tokens["google_drive"]
        self.polygonio_token = tokens["polygonio"]
        
    def get_ohlc_data(self, symbol: str, *, start: datetime, end: datetime, multiplier = 1, measurement = "minute") -> pd.DataFrame:
        """
        Gets requested historical data and returns as a Pandas dataframe.
        Historical data obtained from polygon.io api for the first time is saved locally in the root data folder 
        """
        data_folder = os.path.join(os.environ["workspace"], "data", "raw")
        time_delta = get_time_delta(multiplier, measurement)

        def get_data_with_file_interval(file_interval: str) -> pd.DataFrame:
            folder = os.path.join(data_folder, file_interval)
            os.makedirs(folder, exist_ok=True)
            
            adjusted_start = start if file_interval == "monthly" else start.replace(month = 1)
            adjusted_start = adjusted_start.replace(day = 1)
            adjusted_end = end if file_interval == "monthly" else end.replace(month = 12)
            adjusted_end = adjusted_end.replace(day = get_last_day_of_month(end))
            
            # Process one month at a time
            df = pd.DataFrame()
            range_start = adjusted_start
            range_end = datetime.min # Will set later
            while range_end != adjusted_end and range_end < datetime.now():
                range_end = range_start if file_interval == "monthly" else range_start.replace(month = 12)
                range_end = range_end.replace(day = get_last_day_of_month(range_end))
                file_path = os.path.join(folder, f"{symbol}-{multiplier}-{measurement}-{range_start.date()}-to-{range_end.date()}.csv")
                try:
                    range_df = pd.read_csv(file_path)
                    df = pd.concat([df, range_df])
                    df.reset_index(inplace=True, drop=True)
                    print(range_df)
                    print("Read from csv file")
                except:
                    # File did not exist; use polygon.io API then save the file
                    res = self._get_from_api(symbol, range_start, range_end, multiplier, measurement)
                    range_df = pd.DataFrame.from_dict(res["results"])
                    df = pd.concat([df, range_df])
                    df.reset_index(inplace=True, drop=True)
                    print(range_df)
                    print("Obtained from polygon.io")

                    range_df.to_csv(file_path, index = False)
                    if IS_FREE_TIER:
                        print("On polygon.io free tier - only 5 requests max per minute; sleeping...")
                        time.sleep(60 / 5) # 5 requests per minute

                # 35 days is enough to always set to next month, 366 enough to always set to next year
                delta_days = 35 if file_interval == "monthly" else 366
                range_start = (range_start + timedelta(days = delta_days)).replace(day = 1)
            return df

        if time_delta < timedelta(hours = 1):
            return get_data_with_file_interval("monthly")
        else:
            return get_data_with_file_interval("yearly")
                    
    
    def _get_from_api(self, symbol: str, start: datetime, end: datetime, multiplier = 1, measurement = "minute") -> dict:
        res: dict = requests.get(
            f"https://api.polygon.io/v2/aggs/ticker/{format_symbol_for_api(symbol)}/range/{multiplier}/{measurement}/{start.date()}/{end.date()}?adjusted=true&sort=asc&limit=50000",
            headers={ "Authorization": f"Bearer {self.polygonio_token}" }
        ).json()
        return res

    @staticmethod
    def _create_folder(service: Any, folder_path: str) -> str:
        """
        Creates folder. If exists, leaves it.
        Returns the folder ID
        """
        folder_path = os.path.normpath(folder_path)
        folders = folder_path.split(os.sep)
        parent_id = "root"
        for folder in folders:
            metadata = {
                "name": folder,
                "mimeType": FOLDER_MIME_TYPE,
                "parents": [parent_id]
            }

            page_token = None
            while True:   
                response = service.files().list(q=f"mimeType = '{FOLDER_MIME_TYPE}' and name = '{folder}' and '{parent_id}' in parents", spaces="drive", fields="nextPageToken, files(id, name)", pageToken=page_token).execute()
                files = response.get("files", [])
                if len(files) != 0:
                    parent_id = files[0].get("id")
                    break
                page_token = response.get("nextPageToken", None)
                if page_token is None:
                    file = service.files().create(body=metadata, fields="id").execute()
                    parent_id = file.get("id")
                    break

        return parent_id

    @staticmethod
    def _get_drive_service():
        SCOPES = ['https://www.googleapis.com/auth/drive']
        KEY_FILE_LOCATION = os.path.join(os.environ["workspace"], "tokens", "google_drive_credentials.json")
        if not os.path.isfile(KEY_FILE_LOCATION):
            raise Exception(f"Please download the service account (credential) keys file from Google Cloud Console and place it in {KEY_FILE_LOCATION}")

        # Any is returned from build (it is constructed dynamically), so no intellisense here big man.
        # See https://developers.google.com/drive/api/v3/reference/files
        creds = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE_LOCATION, SCOPES)
        return build("drive", "v3", credentials=creds)

    def _delete_all_drive_items(service: Any):
        response = service.files().list(spaces="drive", fields="files(id)").execute()
        ids = [x["id"] for x in response["files"]]
        for ID in ids:
            service.files().delete(fileId=ID).execute()
        
    def _print_all_drive_items(service: Any):
        response = service.files().list(spaces="drive", fields="files(id, name, mimeType, parents)").execute()
        print(json.dumps(response, indent=2))

    def upload_to_drive(self, filepath: str, drive_filepath: str) -> str:
        """
        Returns the id of the file uploaded to google drive
        If a file already exists with the same name, a new file is created with same name but different id
        """

        service = DataUpdater._get_drive_service()

        folder_id = DataUpdater._create_folder(service, os.path.dirname(drive_filepath))
        
        mime_type: str = mimetypes.guess_type(filepath)[0]
        metadata = {
            "name": os.path.basename(drive_filepath),
            "mimeType": mime_type,
            "parents": [folder_id]
        }
        media = MediaFileUpload(filepath, mimetype=mime_type, resumable=True)
        file = service.files().create(body=metadata, media_body=media, fields="id").execute()

        return file.get("id")
