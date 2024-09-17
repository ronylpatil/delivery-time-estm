# spliting & preprocessing
import pathlib
import yaml
import pandas as pd
from src.logger import infologger
from sklearn.model_selection import train_test_split


# load data from local
def load_data(input_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        infologger.error(f"unable to load data [check load_data()]. exc: {e}")
    else:
        infologger.info(f"data loaded successfully [source: {input_path}]")
        return df


# split the data
def split(
    df: pd.DataFrame, holdout_split: float, unit_test_split: float, seed: int
) -> pd.DataFrame:
    try:
        train, holdout = train_test_split(
            df, test_size=holdout_split, random_state=seed
        )
        test, unit_test = train_test_split(
            holdout, test_size=unit_test_split, random_state=seed
        )
    except Exception as e:
        infologger.error(f"unable to split the data [check split_data()]. exc: {e}")
    else:
        infologger.info(
            f"train-holdout splited successfully [holdout_split: {holdout_split}, seed: {seed}]"
        )
        infologger.info(
            f"test-unit_test splited successfully [unit_test_split: {unit_test_split}, seed: {seed}]"
        )
        return train, test, unit_test


# save the data
def save_data(
    train_test_path: str,
    unit_test_path: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    unit_test: pd.DataFrame,
    train_filename: str = "train",
    test_filename: str = "test",
    unit_test_filename: str = "unit_test",
) -> None:
    try:
        train.to_csv(f"{train_test_path}/{train_filename}.csv", index=False)
        test.to_csv(f"{train_test_path}/{test_filename}.csv", index=False)
        unit_test.to_csv(f"{unit_test_path}/{unit_test_filename}.csv", index=False)
    except Exception as e:
        infologger.error(f"unable to save the data [check save_data()]. exc: {e}")
    else:
        infologger.info(f"training data saved successfully [path: {train_test_path}]")
        infologger.info(f"test data saved successfully [path: {train_test_path}]")
        infologger.info(f"unit_test data saved successfully [path: {unit_test_path}]")


if __name__ == "__main__":
    infologger.info("executing make_dataset.py as __main__")

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(home_dir + "/params.yaml"))["make_dataset"]
    input_path = f"{home_dir}/{params['input_path']}"

    df = load_data(input_path=input_path)
    train, test, unit_test = split(
        df,
        holdout_split=params["holdout_split"],
        unit_test_split=params["unit_test_split"],
        seed=params["seed"],
    )
    
    save_data(f"{home_dir}/{params["train_test_path"]}", f"{home_dir}/{params["unit_test_path"]}", train=train, test=test, unit_test=unit_test)
