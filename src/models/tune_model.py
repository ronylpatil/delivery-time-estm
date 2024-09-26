# implement script to pull latest best performance wise model from mlflow server [use MLflow API]
import yaml
import optuna  # type: ignore
import mlflow
import pathlib
import dagshub
import pandas as pd
from xgboost import XGBRegressor  # type: ignore
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from src.models.train_model import adj_r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger import infologger
from src.models.train_model import save_model


def objective_gbm(trial) -> float:
    # Suggest values for the hyperparameters
    loss = trial.suggest_categorical("loss", ["squared_error", "absolute_error"])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, step=0.01)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    min_samples_split = trial.suggest_float("min_samples_split", 0.1, 0.9, step=0.1)
    min_weight_fraction_leaf = trial.suggest_float(
        "min_weight_fraction_leaf", 0.0, 0.5, step=0.1
    )
    max_depth = trial.suggest_int("max_depth", 3, 8)
    max_features = trial.suggest_float("max_features", 0.1, 0.9, step=0.1)
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 10, 70)

    # Create the RandomForestClassifier with suggested hyperparameters
    model = GradientBoostingRegressor(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_depth=max_depth,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        n_iter_no_change=15,
    )

    # Perform 3-fold cross-validation and calculate accuracy
    score = cross_val_score(model, x_train, y_train, cv=5, scoring="r2").mean()

    return score  # Return the accuracy score for Optuna to maximize


def objective_xgb(trial) -> float:
    # Suggest values for the hyperparameters
    booster = trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"])
    eta = trial.suggest_float("eta", 0.1, 0.3, step=0.01)
    gamma = trial.suggest_int("gamma", 30, 400)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    # lambda = trial.suggest_float("max_features", 0.1, 0.9, step=0.1)
    alpha = trial.suggest_int("alpha", 0, 500)
    tree_method = trial.suggest_categorical(
        "tree_method", ["auto", "exact", "approx", "hist"]
    )

    # Create the RandomForestClassifier with suggested hyperparameters
    model = XGBRegressor(
        booster=booster,  # {gbtree, gblinear, dart}
        eta=eta,  # {learning rate}
        gamma=gamma,  # {}
        max_depth=max_depth,  # {}
        # 'lambda' = 1,
        alpha=alpha,
        tree_method=tree_method,
    )

    # Perform 3-fold cross-validation and calculate accuracy
    score = cross_val_score(model, x_train, y_train, cv=5, scoring="r2").mean()

    return score  # Return the accuracy score for Optuna to maximize


if __name__ == "__main__":

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent.as_posix()

    params_obj = yaml.safe_load(open(f"{home_dir}/params.yaml"))
    params = params_obj["tune_model"]
    model_dir = f"{home_dir}/models"
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

    # dagshub-mlflow setup
    dagshub.init(
        repo_owner=params_obj["mlflow"]["repo_owner"],
        repo_name=params_obj["mlflow"]["repo_name"],
        mlflow=True,
    )

    # Create a study object and optimize the objective function
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler()
    )  # We aim to maximize accuracy
    study.optimize(objective_xgb, n_trials=params["n_trials"])

    # training model with optimized hyperparameters
    best_model = XGBRegressor(**study.best_trial.params)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)

    mae, mse, r2_, adj_r2 = (
        mean_absolute_error(y_test, y_pred),
        mean_squared_error(y_test, y_pred),
        r2_score(y_test, y_pred),
        adj_r2_score(r2_, x_train.shape[0], x_train.shape[1]),
    )

    # setting MLflow
    mlflow.set_experiment("DTE [Fine Tunning XGB]")
    experiment_description = (
        "Tunning xgboost regressor."  # adding experiment description
    )
    mlflow.set_experiment_tag("mlflow.note.content", experiment_description)

    with mlflow.start_run(description="Tunning XGBRegressor - ronil"):
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metrics(
            {
                "mae": round(mae, 3),
                "mse": round(mse, 3),
                "r2_score": round(r2_, 3),
                "adj_r2": round(adj_r2, 3),
            }
        )

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

        mlflow.sklearn.log_model(best_model, "tunned xgbR", signature=signature)
        mlflow.set_tag("developer", "ronil")
        mlflow.set_tag("model", "xgbR")
        infologger.info("Experiment tracked successfully.")

        save_model(
            model=best_model, model_dir=model_dir, model_name=params["model_name"]
        )
