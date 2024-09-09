# data cleaning

import yaml
import pathlib
import inspect
import numpy as np
import pandas as pd
from src import logger
from src.data.make_dataset import load_data
from src.logger import infologger


def cleaning(df: pd.DataFrame) -> pd.DataFrame:

    try:
        # drop ID & Delivery_pperson_id
        df.drop(columns=["ID", "Delivery_person_ID"], inplace=True)

        # rename columns
        df.rename(
            columns={
                "Delivery_person_Age": "Age",
                "Delivery_person_Ratings": "Ratings",
            },
            inplace=True,
        )

        # change datatype of order date column
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")

        # data type formating & fixing col values
        df["Time_Orderd"] = df["Time_Orderd"].replace("NaN ", np.nan)
        df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S").dt.time

        # fixing datatype
        df["Time_Order_picked"] = pd.to_datetime(
            df["Time_Order_picked"], format="%H:%M:%S"
        ).dt.time

        # changing datatype
        df["Vehicle_condition"] = df["Vehicle_condition"].astype("category")

        # fix extra spaces
        df["City"] = list(map(lambda x: x.strip(), df["City"]))

        # fix 'NaN' with np.nan
        df["City"] = df["City"].replace("NaN", np.nan)

        # fix the extra space
        df["Festival"] = list(map(lambda x: x.strip(), df["Festival"]))

        # fix 'NaN' with np.nan
        df["Festival"] = df["Festival"].replace("NaN", np.nan)

        # fix 'NaN ' with np.nan
        df["multiple_deliveries"] = df["multiple_deliveries"].replace("NaN ", np.nan)

        # fixing datatype
        df["multiple_deliveries"] = df["multiple_deliveries"].astype("Int64")

        # as we dont have any clear explanation we'll convert 1,2,3 as 1 which means multiple deliveries
        # and 0 means single delivery
        df["multiple_deliveries"] = df["multiple_deliveries"].apply(
            lambda x: x if x == 0 or np.isnan(x) else 1
        )

        # fix the extra space
        df["Type_of_vehicle"] = list(map(lambda x: x.strip(), df["Type_of_vehicle"]))

        # fix the extra space
        df["Type_of_order"] = list(map(lambda x: x.strip(), df["Type_of_order"]))

        # fix the extra space
        df["Road_traffic_density"] = list(
            map(lambda x: x.strip(), df["Road_traffic_density"])
        )

        # fix 'NaN' with np.nan
        df["Road_traffic_density"] = df["Road_traffic_density"].replace("NaN", np.nan)

        # fixing data type
        df["Ratings"] = df["Ratings"].astype("float")

        # fixing NaN
        df["Age"] = df["Age"].replace("NaN ", np.nan)

        # fixing data type
        df["Age"] = df["Age"].astype("Int64")

        # fixing category names
        df["Weatherconditions"] = df["Weatherconditions"].apply(
            lambda x: x.split(" ")[-1]
        )

        # fixing NaN
        df["Weatherconditions"] = df["Weatherconditions"].replace("NaN", np.nan)

    except Exception as e:
        infologger.error(
            f"Exception caught while cleaning the data. Exc: {e}.\n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
        )
    else:
        infologger.info("Data cleaning completed.")
        return df


def impute_nan(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    try:
        # replace 6 ratting with median
        rating__fillna = train["Ratings"].median()
        train["Ratings"] = train["Ratings"].replace(6, rating__fillna)
        test["Ratings"] = test["Ratings"].replace(6, rating__fillna)

        # NaN replaced by mode
        city__fillna = train["City"].value_counts().index[0]
        train["City"] = train["City"].fillna(value=city__fillna)
        test["City"] = test["City"].fillna(value=city__fillna)

        # NaN replaced by mode of column
        featival__fillna = train["Festival"].value_counts().index[0]
        train["Festival"] = train["Festival"].fillna(value=featival__fillna)
        test["Festival"] = test["Festival"].fillna(value=featival__fillna)

        # filling missing values with mode of col
        multiple_deliveries__fillna = (
            train["multiple_deliveries"].value_counts().index[0]
        )
        train["multiple_deliveries"] = train["multiple_deliveries"].fillna(
            value=multiple_deliveries__fillna
        )
        test["multiple_deliveries"] = test["multiple_deliveries"].fillna(
            value=multiple_deliveries__fillna
        )

        # filling missing with mode of col
        road_traffic_density__fillna = train["Road_traffic_density"].mode()[0]
        train["Road_traffic_density"] = train["Road_traffic_density"].fillna(
            value=road_traffic_density__fillna
        )
        test["Road_traffic_density"] = test["Road_traffic_density"].fillna(
            value=road_traffic_density__fillna
        )

        # fillign missing with mode of col
        weatherconditions__fillna = train["Weatherconditions"].mode()[0]
        train["Weatherconditions"] = train["Weatherconditions"].fillna(
            value=weatherconditions__fillna
        )
        test["Weatherconditions"] = test["Weatherconditions"].fillna(
            value=weatherconditions__fillna
        )
    except Exception as e:
        infologger.error(
            f"Exception caught while imputing NaN. Exc: {e}.\n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
        )
    else:
        infologger.info("Missing values handled perfectly.")
        return train, test


if __name__ == "__main__":
    infologger.info("executing data_cleaning.py as __main__")

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params = yaml.safe_load(open(home_dir + "/params.yaml"))["build_features"]
    # input_path = f"{home_dir}/{params['input_path__train']}"

    # get train, test file path from params.yaml
    train = load_data(f"{home_dir}/{params['input_path__train']}")
    test = load_data(f"{home_dir}/{params['input_path__test']}")

    # cleaning teaning and test set
    clean_train = cleaning(train)
    clean_test = cleaning(test)

    # handling missing values
    final_train, final_test = impute_nan(train=clean_train, test=clean_test)
