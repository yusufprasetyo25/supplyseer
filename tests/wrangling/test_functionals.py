import pandas as pd
import polars as pl

from src.supplyseer.wrangling.functionals import fill_missing_dates


def test_missing_dates():

    data = {
    "date": ["2021-01-01", "2021-01-03", "2021-01-05", "2021-01-07", "2021-01-09", "2021-01-11"],
    "country": ["USA", "USA", "USA", "USA", "USA", "USA"],
    "location": ["New York", "New York", "New York", "New York", "New York", "New York"],
    "no_of_sold_pcs_sum": [100, 150, 200, 250, 300, 350],
    }

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])

    df = fill_missing_dates(df, time_column="date", period_fill="1d", group_by_columns=["country", "location"], keep_order=True)

    df_pl = pl.from_pandas(df)
    df_pl = fill_missing_dates(df_pl, time_column="date", period_fill="1d", group_by_columns=["country", "location"], keep_order=True)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_pl, pl.DataFrame)
