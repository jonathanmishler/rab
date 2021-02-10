import os
import json

import numpy as np
import pandas as pd
import requests

from . import cleaning_pipeline


def download_rab() -> dict:
    """ Retrieves the JSON data for the Registro Aeronáutico Brasileiro (RAB) 
    list of aircraft
    """
    # JSON Endpoint
    url = "https://sistemas.anac.gov.br/dadosabertos/Aeronaves/RAB/dados_aeronaves.json"
    with requests.session() as client:
        resp = client.get(url)
        resp.raise_for_status()
    return resp.json()


def get_rab_data(update: bool = False) -> pd.DataFrame:
    """ Retreives and cleans the Registro Aeronáutico Brasileiro (RAB) 
    Parameters:
        update (bool): Set to True if you want to update the RAB if the file already exists
    """
    # Check if the raw file already exists or if we want to update the current version
    rab_json_filename = "raw_rab.json"
    clean_rab_filename = "rab.pkl"
    if os.path.isfile(clean_rab_filename) and not update:
        print("Registro Aeronáutico Brasileiro (RAB) has already been cleaned")
        return pd.read_pickle(clean_rab_filename)
    elif os.path.isfile(rab_json_filename) and not update:
        print("Registro Aeronáutico Brasileiro (RAB) has already been downloaded")
        with open(rab_json_filename, "r") as file:
            rab_json = json.load(file)
    else:
        print(
            "Downloanding the Registro Aeronáutico Brasileiro (RAB) from Brazilian ANAC"
        )
        rab_json = download_rab()
        with open("raw_rab.json", "w") as file:
            json.dump(rab_json, file)

    df = pd.DataFrame(rab_json)

    if not cleaning_pipeline.check_raw_data(df):
        print(f"BAD COLUMNS: Data is not as expected ({df.columns})")
        return None

    # Pipline for the entire RAB
    df = (
        df.pipe(cleaning_pipeline.rename_columns)
        .drop_duplicates(subset=['tail_number'])
        .pipe(cleaning_pipeline.format_dates, cols=["exp_date_ca", "exp_date_iam"])
        .pipe(
            cleaning_pipeline.str_to_int,
            cols=["year_mfg", "min_crew_size", "max_passengers", "seats"],
        )
        .pipe(cleaning_pipeline.wgt_convert, cols=["max_takeoff_wgt"])
        .pipe(cleaning_pipeline.aircraft_age)
        .pipe(cleaning_pipeline.tax_id)
        .pipe(cleaning_pipeline.owned_and_operated)
    )

    # Pipline for Ag Aircraft
    df = df.pipe(cleaning_pipeline.ag_aircraft).pipe(cleaning_pipeline.icao_engine)

    # Save the cleaned RAB
    df.to_pickle(clean_rab_filename)
    df.to_csv("rab.csv", index=False)

    return df
