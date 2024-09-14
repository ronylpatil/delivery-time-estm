# try non-para ML algo [Bagging & Boosting] as many of col are cat in nature
# mainly DT, Bagging, RF, GB, XGB, and Stacking
# try with/without feature extraction (PCA)

# take input which model you want to train, include 'all' option
# used mlflow for tracking [integrated with dagshub]


import yaml
import mlflow
import joblib
import dagshub
import inspect
import pathlib
import pandas as pd
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from src.logger import infologger


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_to_train: str,
    model_dir: str,
) -> None:

    def adj_r2_score(r2_score: float, no_of_samples: int, no_of_features: int) -> float:
        return 1 - (
            (1 - r2_score)
            * ((no_of_samples - 1) / (no_of_samples - no_of_features - 1))
        )

    if model_to_train == "dt" or model_to_train == "all":

        try:
            model_dt = DecisionTreeRegressor(**params["decision_tree"])
            model_dt.fit(x_train, y_train)
            infologger.info("DecisionTreeRegressor fitted successfully.")
        except Exception as e:
            infologger.error(
                f"Failed to initilize DecisionTreeRegressor. [Error: {e}]. \n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
            )

        y_pred = model_dt.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2_ = r2_score(y_test, y_pred)
        adj_r2 = adj_r2_score(r2_, x_train.shape[0], x_train.shape[1])

        # setting MLflow
        mlflow.set_experiment("DTE [DecisionTree]")
        experiment_description = (
            "Training decision tree regressor."  # adding experiment description
        )
        mlflow.set_experiment_tag("mlflow.note.content", experiment_description)

        with mlflow.start_run(description="Training DecisionTreeRegressor - ronil"):
            mlflow.log_params(params["decision_tree"])
            mlflow.log_metrics(
                {"mae": round(mae, 3), "mse": round(mse, 3), "r2_score": round(r2_, 3), "adj_r2": round(adj_r2, 3)}
            )
            # mlflow.log_artifact()

            mlflow.log_artifact(__file__)  # loging code with mlflow

            # custom model signature
            input_schema = Schema(
                [
                    ColSpec("integer", "Age"),
                    ColSpec("float", "Ratings"),
                    ColSpec("integer", "Weatherconditions"),
                    ColSpec("integer", "Road_traffic_density"),
                    ColSpec("integer", "Vehicle_condition"),
                    ColSpec("integer", "Type_of_order"),
                    ColSpec("integer", "Type_of_vehicle"),
                    ColSpec("integer", "multiple_deliveries"),
                    ColSpec("integer", "Festival"),
                    ColSpec("integer", "City"),
                    ColSpec("float", "haversine_dist"),
                    ColSpec("float", "estm_time"),
                    ColSpec("float", "time_lag"),
                    ColSpec("float", "hour"),
                    ColSpec("integer", "day"),
                    ColSpec("integer", "is_weekend"),
                    ColSpec("integer", "is_rush"),
                ]
            )

            # Define a custom output schema
            output_schema = Schema([ColSpec("float", "Time_taken(min)")])

            # Create a signature
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.sklearn.log_model(model_dt, "decision tree", signature=signature)
            mlflow.set_tag("developer", "ronil")
            mlflow.set_tag("model", "decision tree")

        save_model(model=model_dt, model_dir=model_dir, model_name="model_dt")

    # if model == "rf" or model == "all":
    #     try:
    #         model = RandomForestRegressor(**params["random_forest"])
    #         model.fit(x_train, y)
    #         infologger.info("RandomForestRegressor fitted successfully.")
    #     except Exception as e:
    #         infologger.error(
    #             f"Failed to initilize RandomForestRegressor. [Error: {e}]. \n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
    #         )

    # if model == "gb" or model == "all":
    #     try:
    #         model = GradientBoostingRegressor(**params["gradient_boost"])
    #         model.fit(x_train, y)
    #         infologger.info("GradientBoostingRegressor fitted successfully.")
    #     except Exception as e:
    #         infologger.error(
    #             f"Failed to initilize GradientBoostingRegressor. [Error: {e}]. \n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
    #         )

    # if model == "xgb" or model == "all":
    #     try:
    #         model = XGBRegressor(**params["xgb"])
    #         model.fit(x_train, y)
    #         infologger.info("XGBRegressor fitted successfully.")
    #     except Exception as e:
    #         infologger.error(
    #             f"Failed to initilize XGBRegressor. [Error: {e}]. \n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
    #         )


def save_model(model: BaseEstimator, model_dir: str, model_name: str) -> None:
    try:
        joblib.dump(model, f"{model_dir}/{model_name}.joblib")
        infologger.info(
            f"Model saved successfully [Path: {model_dir}/{model_name}.joblib]"
        )
    except Exception as e:
        infologger.error(
            f"Failed to save the model. [Error: {e}]. \n[File: {pathlib.Path(__file__)}]\n[Method: {inspect.currentframe().f_code.co_name}]"
        )


if __name__ == "__main__":

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params = yaml.safe_load(open(f"{home_dir}/params.yaml"))["train_model"]
    model_dir = f"{home_dir}/models"

    dagshub.init(
        repo_owner=params["mlflow"]["repo_owner"],
        repo_name=params["mlflow"]["repo_name"],
        mlflow=True,
    )

    train_df = pd.read_csv(f"{home_dir}/data/processed/processed_train.csv")
    test_df = pd.read_csv(f"{home_dir}/data/processed/processed_test.csv")
    x_train = train_df.drop(columns=["Time_taken(min)"])
    y_train = train_df["Time_taken(min)"]
    x_test = test_df.drop(columns=["Time_taken(min)"])
    y_test = test_df["Time_taken(min)"]

    train_model(
        x_train,
        y_train,
        x_test,
        y_test,
        model_to_train=params["model_to_train"],
        model_dir=model_dir,
    )
