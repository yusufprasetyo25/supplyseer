import pandas as pd
import polars as pl

def fill_missing_dates(input_data, time_column, period_fill, group_by_columns, keep_order):
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


    
