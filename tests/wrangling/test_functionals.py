import pandas as pd
import polars as pl
import numpy as np

from src.supplyseer.wrangling.functionals import fill_missing_dates, UnivariateTakens


def test_functionals():

    x = np.random.normal(0, 1, 10000).reshape(-1,1)
    # x = np.array([np.nan]*10000).reshape(-1,1) # Uncomment this line to test for NaNs
    takens_rows, takens_columns = x.shape[0] - 3, 3

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

    unitaken = UnivariateTakens()
    takensarr = unitaken.fit_transform(x)



    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_pl, pl.DataFrame)
    assert takensarr.shape == (takens_rows, takens_columns), "Takens embedding array shape is not correct. Did you change 'univariate_takens()' function?"
    assert np.isnan(takensarr).sum() == 0, "NaNs are present in the Takens embedding array. Please check the univarate_takens function"

