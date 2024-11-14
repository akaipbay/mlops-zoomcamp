import pandas as pd
import pytest
from datetime import datetime, timedelta

from homework_solution import batch

def test_prepare_data():
    # Test input
    data = {
        "tpep_pickup_datetime": [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 10, 10),
            datetime(2024, 1, 1, 10, 30),
        ],
        "tpep_dropoff_datetime": [
            datetime(2024, 1, 1, 10, 30),
            datetime(2024, 1, 1, 10, 40),
            datetime(2024, 1, 1, 11, 45),
        ],
        "passenger_count": [1, None, 3],
        "payment_type": [None, 2, None],
    }
    df = pd.DataFrame(data)
    categorical_columns = ["passenger_count", "payment_type"]

    # Run the function
    result_df = batch.prepare_data(df, categorical_columns)

    # Check duration calculation
    expected_durations = [30, 30, 75]  # in minutes
    assert (result_df["duration"].values[:2] == expected_durations[:2]).all()

    # Check duration filtering
    assert result_df["duration"].between(1, 60).all(), "Filtered durations not within range."

    # Check if categorical columns are converted to strings and NaNs handled as -1
    assert result_df["passenger_count"].dtype == object
    assert result_df["payment_type"].dtype == object
    assert (result_df["passenger_count"] == ["1", "-1"]).all()
    assert (result_df["payment_type"] == ["-1", "2"]).all()

    # Check if rows with duration outside 1-60 min are excluded
    assert len(result_df) == 2, "Rows with out-of-range durations were not excluded."

