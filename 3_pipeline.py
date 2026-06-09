import os
import mlflow
import pandas as pd
import numpy as np
import mlflow.sklearn
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from mlflow.models.signature import infer_signature
 
import warnings
warnings.filterwarnings("ignore")
 
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
 
# Variables numéricas
num_features = ['work_year', 'remote_ratio']
 
# Variables ordinales (con orden definido)
ord_exp = ['experience_level']
ord_exp_categories = [['EN', 'MI', 'SE', 'EX']]
 
ord_size = ['company_size']
ord_size_categories = [['S', 'M', 'L']]
 
# Variables nominales (sin orden)
nom_features = ['employment_type', 'job_title_grouped', 'company_location_grouped']
 
best_params = {'n_estimators': 394, 'learning_rate': 0.01561986175938785, 'max_depth': 5, 'min_samples_split': 6, 'min_samples_leaf': 1}
mlflow.set_experiment("Data Science salaries")
 
with mlflow.start_run(run_name="final_model_pipeline"):
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('ord_exp', OrdinalEncoder(categories=ord_exp_categories), ord_exp),
        ('ord_size', OrdinalEncoder(categories=ord_size_categories), ord_size),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nom_features),
    ], remainder='drop')
 
    model = GradientBoostingRegressor(
        **best_params,
        random_state=42
    )
 
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
 
    pipeline.fit(X, y)
 
    input_example = X[:1]
    signature = infer_signature(X, pipeline.predict(X))
 
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="pipeline_data_science",
        input_example=input_example,
        signature=signature
    )
 
    # Guardar el pipeline en la misma carpeta que el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    joblib_path = os.path.join(script_dir, "pipeline_data_science.joblib")
    joblib.dump(pipeline, joblib_path)
    print(f"Pipeline guardado en: {joblib_path}")
 
print("Pipeline saved with MLflow.")