import pandas as pd
import banana_grabber
from . import cleaning_pipeline


class Rab:
    URL = "https://sistemas.anac.gov.br/dadosabertos/Aeronaves/RAB/dados_aeronaves.csv"

    def __init__(self, pathname: str = None) -> None:
        if pathname is None:
            self.pathname = "./data/rab"
        self.raw_filepath = self.get_raw()
        self._clean = self.clean_raw()

    def get_raw(self, update: bool = False):
        print("Getting the Registro Aeronáutico Brasileiro (RAB) from Brazilian ANAC")
        return banana_grabber.grab_from_url(self.URL, self.pathname, "raw.csv", update)

    def clean_raw(self) -> pd.DataFrame:
        if self.raw_filepath is None:
            print("Something went wrong downloading raw file")
            df = None
        else:
            print("Cleaning the Raw Registro Aeronáutico Brasileiro (RAB)")
            df = self.cleaning_pipeline(self.raw)
            # Save the cleaned RAB
            df.to_csv(f"{self.pathname}/clean.csv", index=False)
        return df

    @staticmethod
    def cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
        if not cleaning_pipeline.check_raw_data(df):
            print(f"BAD COLUMNS: Data is not as expected ({df.columns})")
            return None

        # Pipline for the entire RAB
        df = (
            df.pipe(cleaning_pipeline.rename_columns)
            .drop_duplicates(subset=["tail_number"])
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

        return df

    @property
    def raw(self):
        return pd.read_csv(self.raw_filepath, sep=";", skiprows=1, dtype="str", encoding='Latin-1')

    @property
    def clean(self):
        return self._clean

    def update(self):
        self.raw_filepath = self.get_raw(update=True)
        self._clean = self.clean_raw()
