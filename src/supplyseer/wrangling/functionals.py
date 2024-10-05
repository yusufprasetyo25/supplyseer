import pandas as pd
import polars as pl

def fill_missing_dates(input_data, time_column, period_fill, group_by_columns, keep_order):
    """
    Function impute missing dates in a time series data with a forward fill strategy.
    Will automatically fill the group by columns with the same value as the previous row.
    Other columns will not be handled by this function. 

    Runs on Polars backend and is an extension of polars.DataFrame.upsample() method.
    Visit for more: https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.upsample.html

    Args:
        input_data: a pandas or polars dataframe
        time_column: the column with the time series data
        period_fill: the frequency to fill the missing dates
        group_by_columns: the columns to group by
        keep_order: whether to maintain the order of the data

    Returns:
        pd.DataFrame or pl.DataFrame: a dataframe with the missing dates filled
    """

    # Check the type first, if input is a pandas dataframe, convert it to polars and a dummy helper
    dummy_pandas, dummy_polars = False, False

    if isinstance(input_data, pd.DataFrame):
        input_data = pl.from_pandas(input_data).clone()
        dummy_pandas = True

    if isinstance(input_data, pl.DataFrame):
        input_data = input_data.clone()
        dummy_polars = True

        if input_data[time_column].dtype == pl.String:
            input_data = input_data.with_columns(time_column = pl.col(time_column).str.to_date())
    

    if input_data[time_column].dtype not in [pl.Date, pl.Datetime]:
        raise TypeError(f"The '{time_column}' column must be of type Date or Datetime")
    
    # Sort the dataframe by the time column
    input_data = input_data.set_sorted(time_column)

    # Upsample with a filling strategy
    input_data = input_data.upsample(
        time_column=time_column, every=period_fill, group_by=group_by_columns, maintain_order=keep_order
    )

    # Fill the missing values by group_by_columns with a forward fill strategy
    # but it must be an iterative process where it depends on the number of columns given as argument
    # such that it adds only 1 column if input is only 1 column and so on
    input_data = input_data.with_columns(
        [pl.col(col).forward_fill() for col in group_by_columns]
    )

    if dummy_pandas:
        return input_data.to_pandas()
    
    return input_data


    
