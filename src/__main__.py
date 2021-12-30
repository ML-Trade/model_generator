from Model import RNN
from os import environ, path
from Data import DataUpdater

def main():
    file_path = path.dirname(__file__)
    workspace_dir = path.join(file_path, "..")
    environ["workspace"] = path.realpath(workspace_dir)
    print(environ["workspace"])

    data_updater = DataUpdater()
    print(f"GitHub Token: {data_updater.github_token}")
    print(f"Google Drive Token: {data_updater.google_drive_token}")
    print(f"Polygon.io Token: {data_updater.polygonio_token}")

if __name__ == "__main__":
    main()