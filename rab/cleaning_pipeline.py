""" Data cleaning Pipline functions for the RAB """
import re
import datetime
from typing import Optional, List
import collections

import numpy as np
import pandas as pd

from . import brazil_tax_id


COLUMN_MAP_JSON = {
    "MARCA": "tail_number",  # Brazil registration number
    "PROPRIETARIO": "owner_customer_name",
    "OUTROSPROPRIETARIOS": "owner_other",
    "SGUF": "owner_state",
    "CPFCNPJ": "owner_tax_id",
    "NMOPERADOR": "operator_customer_name",
    "OUTROSOPERADORES": "operator_other",
    "UFOPERADOR": "operator_state",
    "CPFCGC": "operator_tax_id",
    "NRCERTMATRICULA": "certifate_num",  # Not sure what certicate number this is referening to "Número dos Certificados (CM - CA)"
    "NRSERIE": "serial",  # Serial number of the aircraft
    "CDCATEGORIA": "opertion_type",  # This field appears to be the Brazilan opertion activity category for the aircraft
    "CDTIPO": "pilot_license_type",  # This field appears to be the pilot license needed to fly this aircraft
    "DSMODELO": "model",  # Model number
    "NMFABRICANTE": "mfg",  # Manufacture name
    "CDCLS": "icao_type_desc",  # ICAO type description, ie: L1T = Land Plane, 1 engine, turboprop
    "NRPMD": "max_takeoff_wgt",  # Max Takeoff Weight (kg)
    "CDTIPOICAO": "icao_type_code",  # ICAO type deignator code, but this is inconsistent and sometimes references the pilot license type
    "NRTRIPULACAOMIN": "min_crew_size",  # Minimum amount of people to operate the aircraft
    "NRPASSAGEIROSMAX": "max_passengers",  # Max number of passengers (not including crew)
    "NRASSENTOS": "seats",  # Number of seats in aircraft
    "NRANOFABRICACAO": "year_mfg",  # Date of manufacture
    "DTVALIDADEIAM": "exp_date_iam",  # Expiration date of the Annual Maintenance Inspection with ANAC
    "DTVALIDADECA": "exp_date_ca",  # Expiration of the Certificate of Airworthiness with ANAC
    "DTCANC": "unk_dtcanc",
    "DSMOTIVOCANC": "unk_dsmotivocanc",
    "CDINTERDICAO": "unk_cdinterdicao",
    "CDMARCANAC1": "unk_cdmarcanac1",
    "CDMARCANAC2": "unk_cdmarcanac2",
    "CDMARCANAC3": "unk_cdmarcanac3",
    "CDMARCAESTRANGEIRA": "foreign_tail_number",  # Previous registration numbers from foriegn countries
    "DSGRAVAME": "unk_dsgravame",
}

COLUMN_MAP_CSV = {
    "MARCA": "tail_number",  # Brazil registration number
    "PROPRIETARIO": "owner_customer_name",
    "OUTROS_PROPRIETARIOS": "owner_other",
    "UF_PROPRIETARIO": "owner_state",
    "CPF_CNPJ_PROPRIETARIO": "owner_tax_id",
    "OPERADOR": "operator_customer_name",
    "OUTROS_OPERADORES": "operator_other",
    "UF_OPERADOR": "operator_state",
    "CPF_CGC_OPERADOR": "operator_tax_id",
    "MATRICULA": "certifate_num",  # Not sure what certicate number this is referening to "Número dos Certificados (CM - CA)"
    "NUM_SERIE": "serial",  # Serial number of the aircraft
    "CATEGORIA": "opertion_type",  # This field appears to be the Brazilan opertion activity category for the aircraft
    "TIPO_CERT": "pilot_license_type",  # This field appears to be the pilot license needed to fly this aircraft
    "MODELO": "model",  # Model number
    "NOME_FABRICANTE": "mfg",  # Manufacture name
    "CLASSE": "icao_type_desc",  # ICAO type description, ie: L1T = Land Plane, 1 engine, turboprop
    "PMD": "max_takeoff_wgt",  # Max Takeoff Weight (kg)
    "TIPO_ICAO": "icao_type_code",  # ICAO type deignator code, but this is inconsistent and sometimes references the pilot license type
    "TRIP_MIN": "min_crew_size",  # Minimum amount of people to operate the aircraft
    "PAX_MAX": "max_passengers",  # Max number of passengers (not including crew)
    "ASSENTOS": "seats",  # Number of seats in aircraft
    "ANO_FABRICACAO": "year_mfg",  # Date of manufacture
    "VAL_CAV": "exp_date_iam",  # Expiration date of the Annual Maintenance Inspection with ANAC
    "VAL_CA": "exp_date_ca",  # Expiration of the Certificate of Airworthiness with ANAC
    "DATA_CANC": "unk_dtcanc",
    "MOTIVO": "unk_dsmotivocanc",
    "CD_INTERDICAO": "unk_cdinterdicao",
    "MARCA_NAC_1": "unk_cdmarcanac1",
    "MARCA_NAC_2": "unk_cdmarcanac2",
    "MARCA_NAC_3": "unk_cdmarcanac3",
    "MARCA_EST": "foreign_tail_number",  # Previous registration numbers from foriegn countries
    "DESCRICAO_DO_GRAVAME": "unk_dsgravame",
}


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ Renames the columns using the COLUMN MAP """
    return df.rename(columns=COLUMN_MAP_CSV)


def check_raw_data(data: pd.DataFrame) -> bool:
    """ Checks the data retrieved from the endpoint that it has the expected columns """
    expected_cols = collections.Counter(list(COLUMN_MAP_CSV.keys()))
    response_cols = collections.Counter(list(data.columns))
    return expected_cols == response_cols


def format_dates(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Converts dates formatted as DDMMYY[YY] to YYYY-MM-DD
    Note:
        If the value is not a number, then it will return the value as is,
        and for this reason, the column remains a string dtype.
    """

    def apply_func(x):
        if x is None:
            return x
        try:
            int(x)
        except ValueError:
            return x
        d = x[0:2]
        m = x[2:4]
        y = x[4:]
        if len(y) == 2:
            if int(y) < 20:
                y = "20" + y
            else:
                y = "19" + y
        return f"{y}-{m}-{d}"

    for col in cols:
        df[col] = df[col].apply(apply_func)

    return df


def str_to_int(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """ Cleans and converts to the new pandas integer type (Int64) """
    for col in cols:
        df[col] = pd.to_numeric(df[col]).astype("Int64")
    return df


def wgt_convert(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """ Cleans the weight in kg to a float all NA values are -1  """
    for col in cols:
        df[col] = (
            df[col]
            .str.replace(r"[^0-9,.]", "")
            .str.replace(",", ".")
            .str.strip()
            .fillna(-1)
            .astype("float")
        )

    return df


def aircraft_age(df: pd.DataFrame) -> pd.DataFrame:
    """ Adds an age column to the dataframe with the age in years of the aircraft """
    # Anything before 1910 needs to be excluded as this is not a valid year
    df.loc[(df["year_mfg"].notna()) & (df["year_mfg"] < 1910), "year_mfg"] = pd.NA
    df["age"] = datetime.datetime.today().year - df["year_mfg"]
    return df


def tax_id_print(x: pd.Series) -> Optional[str]:
    """ Creates the printable string for the tax id """
    if x[0] == "INVALID":
        return x[1]
    elif x[0] == "CNPJ":
        return brazil_tax_id.format_cnpj(x[1])
    elif x[0] == "CPF":
        return brazil_tax_id.format_cpf(x[1])
    else:
        return pd.NA


def tax_id(df: pd.DataFrame) -> pd.DataFrame:
    """All the manuplulations for the tax id in the data
    1. Any tax_id = 0 needs to be set to pd.NA
    2. Add a type field for the tax id for owners and operators
    3. Check if numbers are valid and set each type
    4. Add a column for the correct prinatbale string
    """
    for entity in ["owner", "operator"]:
        # Column names
        id_col = f"{entity}_tax_id"
        type_col = f"{id_col}_type"
        print_col = f"{id_col}_print"

        # set to NA if the string is all 0
        mask_missing_tax_ids = (
            df[id_col].fillna("0").apply(lambda x: sum([int(i) for i in list(x)])) == 0
        )
        df.loc[mask_missing_tax_ids, id_col] = pd.NA

        # Create type column init with INVALID
        df[type_col] = "INVALID"
        # Set NA values as type EMPTY
        df.loc[df[id_col].isna(), type_col] = "EMPTY"

        # Subset of valid CNPJ or CPF
        cnpj = df[id_col].apply(brazil_tax_id.valid_cnpj)
        cpf = df[id_col].apply(brazil_tax_id.valid_cpf)

        # Sets the valid types as either a CPF or CNPJ
        df.loc[cnpj, type_col] = "CNPJ"
        df.loc[cpf, type_col] = "CPF"

        # Creates the printable string for the tax id
        df[print_col] = df[[type_col, id_col]].apply(tax_id_print, axis=1)

    return df


def owned_and_operated(df: pd.DataFrame):
    """ Adds a column which signifies if the owner and operator of the aircraft match """
    df["owned_operated"] = df.apply(
        lambda x: x["owner_tax_id"] == x["operator_tax_id"], axis=1
    ).fillna(False)
    return df


def search_df(df: pd.DataFrame, col: str, regex_pat: str) -> pd.Series:
    """ Creates a series of bools for the regex pattern contained in the column """
    return df[col].str.contains(regex_pat, flags=re.IGNORECASE, na=False)


def bool_df(df: pd.DataFrame, query_list: List[tuple]) -> pd.DataFrame:
    """ Creates a dataframe of bools for each tuple of (column name, regex pattern) """
    return pd.concat(
        [search_df(df, column, pattern) for column, pattern in query_list], axis=1
    )


def ag_aircraft(df: pd.DataFrame) -> pd.DataFrame:
    """ Adds agaircraft column and normalizes manufcature name """
    aircraft = [
        {
            "name": "AG-CAT",
            "normalize": True,
            "query": [
                ("mfg", "(AG){1}[-_\s]*(CAT){1}"),
                ("model", "G{1}[-_\s]*(164){1}A?"),
            ],
            "type": "any",
        },
        {
            "name": "THRUSH AIRCRAFT",
            "normalize": True,
            "query": [("mfg", "(THRUSH)"), ("model", "(S2R-)")],
            "type": "any",
        },
        {
            "name": "CESSNA AIRCRAFT",
            "normalize": True,
            "query": [("mfg", "(CESSNA)"), ("model", "(188){1}")],
            "type": "all",
        },
        {
            "name": "PIPER AIRCRAFT",
            "normalize": False,
            "query": [("model", "(PA){1}[-_\s]*(25){1}")],
            "type": "any",
        },
        {
            "name": "EMBRAER",
            "normalize": True,
            "query": [("model", "(EMB){1}[-_\s]*(2){1}")],
            "type": "any",
        },
        {
            "name": "AIR TRACTOR",
            "normalize": True,
            "query": [
                ("mfg", "AIR TRACTOR"),
                ("model", "(AT){1}[-_\s](40|50|60|80){1}"),
            ],
            "type": "any",
        },
    ]

    ag_aircraft_selection = list()

    for mfg in aircraft:
        if mfg["type"] == "any":
            selection = bool_df(df, mfg["query"]).any(axis=1)
        elif mfg["type"] == "all":
            selection = bool_df(df, mfg["query"]).all(axis=1)

        if mfg["normalize"]:
            df.loc[selection, "mfg"] = mfg["name"]

        ag_aircraft_selection.append(selection)
    ag_aircraft_selection = pd.concat(ag_aircraft_selection, axis=1).any(axis=1)

    df["agaircraft"] = False
    df.loc[ag_aircraft_selection, "agaircraft"] = True

    return df


def icao_engine(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the icao_type_desc to the following for ag aircraft:
    - L1T: Land Plane, Single Engine, Turboprop
    - L1P: Land Plane, Single Engine, Piston
    """
    # make selection for radial Air Tractors
    sel_401 = bool_df(df, [("mfg", "AIR TRACTOR"), ("model", "(401){1}")]).all(axis=1)
    # selection for all turbine ag_aircraft
    sel_turbine_ag = ((df.mfg == "AIR TRACTOR") & -sel_401) | (
        df.mfg == "THRUSH AIRCRAFT"
    )
    # Set ICAO Type for Turbine Ag Aircraft
    df.loc[(sel_turbine_ag), "icao_type_desc"] = "L1T"
    # Set ICAO Type for Piston Ag Aircraft
    df.loc[(-sel_turbine_ag & df["agaircraft"]), "icao_type_desc"] = "L1P"

    return df
