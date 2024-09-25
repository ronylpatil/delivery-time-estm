# implement script to pull latest best performance wise model from mlflow server [use MLflow API]
# save it in ./models/production
# use optuna with rf, gb, xgb
import yaml
import optuna  # type: ignore
import pathlib
import numpy as np
import pandas as pd
from functools import partial
from hyperopt.pyll import scope  # type: ignore
from src.models.train_model import adj_r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Define the objective function
def objective(trial):
    # Suggest values for the hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 250)
    criterion = trial.suggest_categorical(
        "criterion", ["squared_error", "absolute_error"]
    )
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 15, 50)
    min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.2, 0.7, step=0.1)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    # Create the RandomForestClassifier with suggested hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
    )

    # Perform 3-fold cross-validation and calculate accuracy
    score = cross_val_score(model, x_train, y_train, cv=5, scoring="r2").mean()

    return score  # Return the accuracy score for Optuna to maximize


if __name__ == "__main__":

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params_obj = yaml.safe_load(open(f"{home_dir}/params.yaml"))

    train_df = pd.read_csv(
        f"{home_dir}{params_obj['feature_engineering']['export_path']}/processed_train.csv"
    )
    test_df = pd.read_csv(
        f"{home_dir}{params_obj['feature_engineering']['export_path']}/processed_test.csv"
    )
    x_train = train_df.drop(columns=params_obj["base"]["target"])
    y_train = train_df[params_obj["base"]["target"]]
    x_test = test_df.drop(columns=params_obj["base"]["target"])
    y_test = test_df[params_obj["base"]["target"]]

    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )  # We aim to maximize accuracy
    study.optimize(
        objective, n_trials=50
    )  # Run 50 trials to find the best hyperparameters

    # Print the best result
    print(f"Best trial accuracy: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")

    best_model = RandomForestRegressor(**study.best_trial.params)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2_ = r2_score(y_test, y_pred)
    adj_r2 = adj_r2_score(r2_, x_train.shape[0], x_train.shape[1])

    print(f"MAE: {mae}\nMSE: {mse}\nR2 Score: {r2_}\nAdj R2: {adj_r2}\n")
