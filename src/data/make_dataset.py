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
        infologger.info(f"unable to load data [check load_data()]. exc: {e}")
    else:
        infologger.info(f"data loaded successfully [source: {input_path}]")
        return df


# split the data
def split(df: pd.DataFrame, test_split: float, seed: int) -> pd.DataFrame:
    try:
        train, test = train_test_split(df, test_size=test_split, random_state=seed)
    except Exception as e:
        infologger.info(f"unable to split the data [check split_data()]. exc: {e}")
    else:
        infologger.info(
            f"data splited successfully [test_split: {test_split}, seed: {seed}]"
        )
        return train, test


# save the data
def save_data(export_path: str, train: pd.DataFrame, test: pd.DataFrame, train_filename: str = 'train', test_filename: str = 'test') -> None:
    try:
        train.to_csv(f"{export_path}/{train_filename}.csv", index=False)
        test.to_csv(f"{export_path}/{test_filename}.csv", index=False)
    except Exception as e:
        infologger.info(f"unable to save the data [check save_data()]. exc: {e}")
    else:
        infologger.info(f"training data saved successfully [path: {export_path}]")
        infologger.info(f"test data saved successfully [path: {export_path}]")


if __name__ == "__main__":
    infologger.info("executing make_dataset.py as __main__")

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()
    params = yaml.safe_load(open(home_dir + "/params.yaml"))["make_dataset"]
    input_path = f"{home_dir}/{params['input_path']}"

    df = load_data(input_path=input_path)
    train, test = split(df, test_split=params["test_split"], seed=params["seed"])
    save_data(params["export_path"], train=train, test=test)
