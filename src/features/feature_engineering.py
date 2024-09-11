# build new feature here
"""
"restaurant - delivery location" latitude & longitude
- Euclidean dist
- haversine dist
- manhattan dist
- estimated delivery time (assume 30 kmph)

time ordered & time order picked
- time lag
- ordered picked time ratio

order date
- hour
- day of week
- month
- quarter
- year
"""

import yaml
import pathlib
import inspect
import numpy as np
import pandas as pd
from src.logger import infologger
from sklearn.preprocessing import LabelEncoder
from src.data.make_dataset import load_data, save_data
from math import radians, sin, cos, sqrt, atan2


# initial preprocessing
def cleaning(df: pd.DataFrame) -> pd.DataFrame:  # [STEP-I]
    """
    Perform initial data preprocessing on given dataframe `df`

    Parameters
    ----------
    df : {ndarray, dataframe}
        Data to be clean.

    Returns
    -------
    df : {ndarray, dataframe}
        Cleaned dataframe.
    """
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


# fill NaN
def impute_nan(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:  # [STEP-II]
    """
    Fill NaN with suitable values

    Parameters
    ----------
    train : {ndarray, dataframe}
            Training dataset
    test : {ndarray, dataframe}
            Test dataset

    Returns
    -------
    df : {ndarray, dataframe}
        Preprocessed DataFrame
    """
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


# calculate Haversine dist
def haversine(df: pd.DataFrame) -> float:
    lat1, lon1, lat2, lon2 = (
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(abs(lat2) - abs(lat1))
    dlon = radians(abs(lon2) - abs(lon1))

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return round(R * c, 3)  # Distance in kilometers


# calculate estimated delivery time assuming 30 km/h
def estimated_delivery_time(df: pd.DataFrame):
    distance_km = df["haversine_dist"]
    return distance_km / 30  # Time in hours


# building new features
def build_features(df: pd.DataFrame) -> pd.DataFrame:  # [STEP-III]
    """
    Build new features from existing ones.

    Parameters
    ----------
    df : {ndarray, dataframe}
        Training/Testing Data

    Returns
    -------
    df : {ndarray, dataframe}
        Feature Engineered Data
    """
    try:
        # calculate haversine dist
        df["haversine_dist"] = df.apply(haversine, axis=1)

        # estimated time (in min)
        df["estm_time"] = df.apply(estimated_delivery_time, axis=1)
        df["estm_time"] = round(df["estm_time"] * 60, 3)

        # get ordered_hour, ordered_min, picked_hour, picked_min
        df["ordered_hour"] = df["Time_Orderd"].apply(lambda x: x.hour)
        df["ordered_min"] = df["Time_Orderd"].apply(lambda x: x.minute)
        df["picked_hour"] = df["Time_Order_picked"].apply(lambda x: x.hour)
        df["picked_min"] = df["Time_Order_picked"].apply(lambda x: x.minute)

        # ese order bhi he jo 12 bje se phle order hue ho or 12 k baad pickup hue ho (in min)
        df["time_lag"] = np.where(
            (df["picked_hour"] - df["ordered_hour"]) < 0,
            (23 + df["picked_hour"] - df["ordered_hour"]) * 60
            + (60 + df["picked_min"] - df["ordered_min"]),
            (df["picked_hour"] - df["ordered_hour"]) * 60
            + (df["picked_min"] - df["ordered_min"]),
        )

        # extract hour
        df["hour"] = df["Time_Orderd"].apply(lambda x: x.hour)

        # get week day
        df["day"] = df["Order_Date"].dt.strftime("%A")

        # is_weekend
        df["is_weekend"] = df["day"].apply(
            lambda x: 1 if x in ["Saturday", "Sunday"] else 0
        )

        # 5. Rush Hour (1 if order time is between 11:00-14:00 or 19:00-23:00, else 0)
        df["is_rush"] = df["hour"].apply(
            lambda x: 1 if (11 <= x <= 14) or (19 <= x <= 23) else 0
        )
    except Exception as e:
        infologger.error(
            f"Exception caught while building features. Exc: {e}.\n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
        )
    else:
        infologger.info(f"Featuring engineering completed.")
        return df


# drop MAR missing values
# there're diff approach to handle MAR, but I preffered droping them
def drop_mar(df: pd.DataFrame) -> pd.DataFrame:  # [STEP-IV]
    """
    Drop co-related missing values.

    Parameters
    ----------
    df : {ndarray, dataframe}
        Training/Testing DataFrame

    Returns
    -------
    df : {ndarray, dataframe}
        Preprocessed DataFrame
    """
    try:
        df.dropna(inplace=True)
    except Exception as e:
        infologger.error(
            f"Exception caught while droping NaN. Exc: {e}.\n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
        )
    else:
        infologger.info(f"NaN dropped successfully.")
        return df


# final preprocessing
def preprocessing(
    train: pd.DataFrame, test: pd.DataFrame, path: pathlib.Path
) -> pd.DataFrame:  # [STEP-V]
    """
    Perform final preprocessing on the data.

    Parameters
    ----------
    train: {ndarray, dataframe}
            Training dataset
    test: {ndarray, dataframe}
            Test dataset
    path: str
          File path where label encoding will stored.

    Returns
    -------
    df : {ndarray, dataframe}
        Preprocessed DataFrame
    """
    try:
        # drop lat/lon col
        # ├── Restaurant_latitude
        # ├── Restaurant_longitude
        # ├── Delivery_location_latitude
        # └── Delivery_location_longitude
        train.drop(
            columns=[
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
            ],
            inplace=True,
        )

        test.drop(
            columns=[
                "Restaurant_latitude",
                "Restaurant_longitude",
                "Delivery_location_latitude",
                "Delivery_location_longitude",
            ],
            inplace=True,
        )

        # drop date/time columns
        # ├── Order_Date
        # ├── Time_Orderd
        # ├── Time_Order_picked
        # ├── picked_min
        # ├── picked_hour
        # ├── ordered_min
        # └── ordered_hour
        train.drop(
            columns=[
                "Order_Date",
                "Time_Orderd",
                "Time_Order_picked",
                "picked_min",
                "ordered_min",
                "picked_hour",
                "ordered_hour",
            ],
            inplace=True,
        )

        test.drop(
            columns=[
                "Order_Date",
                "Time_Orderd",
                "Time_Order_picked",
                "picked_min",
                "ordered_min",
                "picked_hour",
                "ordered_hour",
            ],
            inplace=True,
        )

        # label encoding of cat data
        # ├── Weatherconditions
        # ├── Road_traffic_density
        # ├── City
        # ├── Festival
        # ├── Type_of_vehicle
        # ├── Type_of_order
        # └── Vehicle_condition

        # NOTE: in prod we need to transform user input into numerical so keep the
        # mapping ready.

        if path.exists():
            infologger.info(f"labels exist- [{path}]")
            path.unlink()
            infologger.info(f"labels dropped.")

        with open(path, "a") as f:
            le = LabelEncoder()

            train["Weatherconditions"] = le.fit_transform(train["Weatherconditions"])
            test["Weatherconditions"] = le.transform(test["Weatherconditions"])
            f.write(
                f"Weatherconditions: {dict(zip(le.classes_, map(int, le.transform(le.classes_))))}\n"
            )

            train["Road_traffic_density"] = le.fit_transform(
                train["Road_traffic_density"]
            )
            test["Road_traffic_density"] = le.transform(test["Road_traffic_density"])
            f.write(
                f"Road_traffic_density: {dict(zip(le.classes_, map(int, le.transform(le.classes_))))}\n"
            )

            train["City"] = le.fit_transform(train["City"])
            test["City"] = le.transform(test["City"])
            f.write(
                f"City: {dict(zip(le.classes_, map(int, le.transform(le.classes_))))}\n"
            )

            train["Festival"] = le.fit_transform(train["Festival"])
            test["Festival"] = le.transform(test["Festival"])
            f.write(
                f"Festival: {dict(zip(le.classes_, map(int, le.transform(le.classes_))))}\n"
            )

            train["Type_of_vehicle"] = le.fit_transform(train["Type_of_vehicle"])
            test["Type_of_vehicle"] = le.transform(test["Type_of_vehicle"])
            f.write(
                f"Type_of_vehicle: {dict(zip(le.classes_, map(int, le.transform(le.classes_))))}\n"
            )

            train["Type_of_order"] = le.fit_transform(train["Type_of_order"])
            test["Type_of_order"] = le.transform(test["Type_of_order"])
            f.write(
                f"Type_of_order: {dict(zip(le.classes_, map(int, le.transform(le.classes_))))}\n"
            )

            train["day"] = le.fit_transform(train["day"])
            test["day"] = le.transform(test["day"])
            f.write(
                f"day: {dict(zip(le.classes_, map(int, le.transform(le.classes_))))}\n"
            )

        # fix target variable
        train["Time_taken(min)"] = (
            train["Time_taken(min)"].apply(lambda x: x.split(" ")[-1]).astype("int")
        )
        test["Time_taken(min)"] = (
            test["Time_taken(min)"].apply(lambda x: x.split(" ")[-1]).astype("int")
        )
    except Exception as e:
        infologger.error(
            f"Exception caught while performing final preprocessing. Exc: {e}.\n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
        )
    else:
        infologger.info(f"final preprocessing completed.")
        return train, test


if __name__ == "__main__":
    infologger.info("executing build_features.py as __main__")

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(home_dir + "/params.yaml"))["feature_engineering"]

    # reading train / test dataset path
    input_path__train = f"{home_dir}/{params['input_path__train']}"
    input_path__test = f"{home_dir}/{params['input_path__test']}"

    # loading train / test dataset
    training_data = load_data(input_path=input_path__train)
    test_data = load_data(input_path=input_path__test)

    # STAGE-I
    train_st1 = cleaning(training_data)
    test_st1 = cleaning(test_data)

    # STAGE-II
    train_st2, test_st2 = impute_nan(train_st1, test_st1)

    # STAGE-III
    train_st3 = build_features(train_st2)
    test_st3 = build_features(test_st2)

    # STAGE-IV
    train_st4 = drop_mar(train_st3)
    test_st4 = drop_mar(test_st3)

    # STAGE-V
    path_labels = pathlib.Path(f"{home_dir}/{params['labels_']}")
    train_st5, test_st5 = preprocessing(train_st4, test_st4, path=path_labels)

    # save the data
    export_path = f"{home_dir}/{params["export_path"]}"
    save_data(
        export_path=export_path,
        train=train_st5,
        test=test_st5,
        train_filename=params["filename_train"],
        test_filename=params["filename_test"],
    )
