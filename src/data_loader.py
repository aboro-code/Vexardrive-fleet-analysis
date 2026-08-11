import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "VEXAR_Fleet_Dataset_CANDIDATE_VERSION.xlsx"
)


def load_all(path=DATA_PATH):
    drivers = pd.read_excel(path, sheet_name="Drivers", header=2)
    vehicles = pd.read_excel(path, sheet_name="Vehicles", header=2)
    trips = pd.read_excel(path, sheet_name="Trips", header=2)
    telemetry = pd.read_excel(path, sheet_name="Telemetry", header=2)
    return drivers, vehicles, trips, telemetry
