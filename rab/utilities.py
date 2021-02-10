import datetime
import pandas as pd


def customer_column_map(customer_type: str) -> dict:
    """ Renames the columns for the owner or operator info so they are easier to map to the html template """
    info_columns = ["customer_name", "tax_id", "tax_id_type", "tax_id_print", "other"]
    column_map = {
        c[1]: c[0]
        for c in zip(info_columns, [f"{customer_type}_{col}" for col in info_columns])
    }
    return column_map


def create_customer_df(df: pd.DataFrame, customer_type: str) -> pd.DataFrame:
    """ Modifies the dataframe to just include the specific customer type of 
    owner or operator and drops the other info.  Also, creates a customer_type column.
    """
    if customer_type == "owner":
        drop_type = "operator"
    elif customer_type == "operator":
        drop_type = "owner"
    else:
        print("Not a valid customer_type.  Needs to be either owner or opeartor")
        return df

    df = df.rename(columns=customer_column_map(customer_type))
    customer_column_map(drop_type)
    df = df.drop(customer_column_map(drop_type).keys(), axis=1)
    df["customer_type"] = customer_type
    return df


def melt_rab_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """ Melts the RAB by owner and operator so each type is on their own row per tail number"""
    return pd.concat(
        [create_customer_df(df, cust_type) for cust_type in ["owner", "operator"]]
    )


def is_past_due(date: str):
    """ Checks if a date is past today 
    Note:
        If the date is not ISO format, it will set it to NA 
    """
    if date is None:
        return None
    try:
        return datetime.date.fromisoformat(date) < datetime.date.today()
    except ValueError:
        return None
