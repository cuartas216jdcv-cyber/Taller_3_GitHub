import os
import optuna
import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn
import optuna.visualization as vis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from mlflow.models.signature import infer_signature

data = pd.read_csv('ds_salaries.csv')

# Columnas a eliminar:
# - 'salary' y 'salary_currency': redundante con salary_in_usd (ya está en USD)
# - 'employee_residence': alta correlación con company_location
COLS_DROP = ['salary', 'salary_currency', 'employee_residence']
data = data.drop(columns=COLS_DROP)
data['salary_log'] = np.log1p(data['salary_in_usd']) #se va a utilizar salary log por lo tanto mae y rmse sale en log tambien

# Agrupar job_titles poco frecuentes como 'Other' para reducir cardinalidad
top_jobs = data['job_title'].value_counts()
top_jobs_list = top_jobs[top_jobs >= 10].index.tolist()
data['job_title_grouped'] = data['job_title'].apply(
    lambda x: x if x in top_jobs_list else 'Other'
)

# Agrupar países poco frecuentes
top_countries = data['company_location'].value_counts()
top_countries_list = top_countries[top_countries >= 10].index.tolist()
data['company_location_grouped'] = data['company_location'].apply(
    lambda x: x if x in top_countries_list else 'Other'
)

# Features de entrada
FEATURES = [
    'work_year',
    'experience_level', # ordinal: EN < MI < SE < EX
    'employment_type', # nominal: FT, PT, CT, FL
    'job_title_grouped', # nominal (agrupado)
    'company_location_grouped', # nominal (agrupado)
    'remote_ratio', # numérica: 0, 50, 100
    'company_size', # ordinal: S < M < L
]

TARGET = 'salary_log'  # regresión con log-transform

X = data[FEATURES].copy()
y = data[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Variables numéricas
num_features = ['work_year', 'remote_ratio']

# Variables ordinales (con orden definido)
ord_exp = ['experience_level']
ord_exp_categories = [['EN', 'MI', 'SE', 'EX']]

ord_size = ['company_size']
ord_size_categories = [['S', 'M', 'L']]

# Variables nominales (sin orden)
nom_features = ['employment_type', 'job_title_grouped', 'company_location_grouped']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('ord_exp', OrdinalEncoder(categories=ord_exp_categories), ord_exp),
    ('ord_size', OrdinalEncoder(categories=ord_size_categories), ord_size),
    ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nom_features),
], remainder='drop')

preprocessor.fit(X_train)

X_train_prep = preprocessor.transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

results = []

mlflow.set_experiment("Data Science salaries")

def calculate_metrics(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return r2, mae, rmse

def run_optuna_model(
    model_name,
    model_class,
    objective_func,
    n_trials=50,
    extra_params=None
):

    study = optuna.create_study(direction="minimize")

    with mlflow.start_run(run_name=model_name):

        study.optimize(objective_func, n_trials=n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mae", study.best_value)

        best_params = study.best_params.copy()

        if extra_params:
            best_params.update(extra_params)

        final_model = model_class(**best_params)

        final_model.fit(X_train_prep, y_train)

        y_pred = final_model.predict(X_test_prep)

        final_r2, final_mae, final_rmse = calculate_metrics(
            y_test,
            y_pred
        )

        results.append({
            "model": model_name,
            "mae": final_mae,
            "r2": final_r2,
            "rmse": final_rmse,
            "best_params": best_params
        })

        mlflow.log_metric("final_r2", final_r2)
        mlflow.log_metric("final_mae", final_mae)
        mlflow.log_metric("final_rmse", final_rmse)

        signature = infer_signature(
            X_train_prep,
            final_model.predict(X_train_prep)
        )

        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path=model_name.lower(),
            input_example=X_train.iloc[:1],
            signature=signature
        )

        history_file = f"{model_name}_history.png"
        slice_file = f"{model_name}_slice.png"

        vis.plot_optimization_history(study).write_image(history_file)
        vis.plot_slice(study).write_image(slice_file)

        mlflow.log_artifact(history_file)
        mlflow.log_artifact(slice_file)

        os.remove(history_file)
        os.remove(slice_file)


    return study

def objective_rf(trial): #RandomForestRegressor

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2"]
        )
    }

    model = RandomForestRegressor(
        **params,
        random_state=42
    )

    model.fit(X_train_prep, y_train)

    y_pred = model.predict(X_test_prep)

    _, mae, _ = calculate_metrics(y_test, y_pred)

    with mlflow.start_run(
        nested=True,
        run_name=f"Iteration {trial.number+1}"
    ):
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)

    return mae

def objective_dt(trial): #DecisionTreeRegressor

    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None]
        )
    }

    model = DecisionTreeRegressor(
        **params,
        random_state=42
    )

    model.fit(X_train_prep, y_train)

    y_pred = model.predict(X_test_prep)

    _, mae, _ = calculate_metrics(y_test, y_pred)

    with mlflow.start_run(
        nested=True,
        run_name=f"Iteration {trial.number+1}"
    ):
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)

    return mae

def objective_gb(trial): #GradientBoostingRegressor

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            500
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            10
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            2,
            20
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            1,
            10
        )
    }

    model = GradientBoostingRegressor(
        **params,
        random_state=42
    )

    model.fit(X_train_prep, y_train)

    y_pred = model.predict(X_test_prep)

    _, mae, _ = calculate_metrics(y_test, y_pred)

    with mlflow.start_run(
        nested=True,
        run_name=f"Iteration {trial.number+1}"
    ):
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)

    return mae

def objective_ridge(trial): #Ridge

    params = {
        "alpha": trial.suggest_float(
            "alpha",
            1e-4,
            100,
            log=True
        )
    }

    model = Ridge(**params)

    model.fit(X_train_prep, y_train)

    y_pred = model.predict(X_test_prep)

    _, mae, _ = calculate_metrics(y_test, y_pred)

    with mlflow.start_run(
        nested=True,
        run_name=f"Iteration {trial.number+1}"
    ):
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)

    return mae

def objective_xgb(trial): #XGBRegressor

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            1000
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.3,
            log=True
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        )
    }

    model = XGBRegressor(
        **params,
        random_state=42,
        objective="reg:squarederror",
        eval_metric="rmse"
    )

    model.fit(X_train_prep, y_train)

    y_pred = model.predict(X_test_prep)

    _, mae, _ = calculate_metrics(y_test, y_pred)

    with mlflow.start_run(
        nested=True,
        run_name=f"Iteration {trial.number+1}"
    ):
        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)

    return mae

run_optuna_model("RandomForest",
    RandomForestRegressor,
    objective_rf,
    extra_params={"random_state": 42}
)

run_optuna_model("DecisionTree",
    DecisionTreeRegressor,
    objective_dt,
    extra_params={"random_state": 42}
)

run_optuna_model("GradientBoosting",
    GradientBoostingRegressor,
    objective_gb,
    extra_params={"random_state": 42}
)

run_optuna_model("Ridge",
    Ridge,
    objective_ridge
)

run_optuna_model("XGBoost",
    XGBRegressor,
    objective_xgb,
    extra_params={
        "random_state": 42,
        "objective": "reg:squarederror",
        "eval_metric": "rmse"
    }
)

results_df = pd.DataFrame(results)

print("RESULTADOS FINALES")

print(
    results_df[["model", "mae", "r2", "rmse"]]
    .sort_values("mae")
)

best_model = results_df.loc[
    results_df["mae"].idxmin()
]

print("MEJOR MODELO")

print(f"Modelo: {best_model['model']}")
print(f"MAE: {best_model['mae']:.6f}")
print(f"R2: {best_model['r2']:.6f}")
print(f"RMSE: {best_model['rmse']:.6f}")

print("MEJORES PARÁMETROS")

print(best_model["best_params"])