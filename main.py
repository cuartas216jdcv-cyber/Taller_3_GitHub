from pandas import DataFrame
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException
import numpy as np
import joblib

# Cargar modelo desde MLflow
model = joblib.load("pipeline_data_science.joblib")

app = FastAPI()

class DataPredict(BaseModel):
    work_year: int
    experience_level: str
    employment_type: str
    job_title_grouped: str
    company_location_grouped: str
    remote_ratio: int
    company_size: str


@app.post("/predict")
def predict(request: DataPredict):

    try:

        df_data = DataFrame([{
            "work_year": request.work_year,
            "experience_level": request.experience_level,
            "employment_type": request.employment_type,
            "job_title_grouped": request.job_title_grouped,
            "company_location_grouped": request.company_location_grouped,
            "remote_ratio": request.remote_ratio,
            "company_size": request.company_size
        }])

        pred_log = model.predict(df_data)[0]

        pred_salary = float(np.expm1(pred_log))

        return {
            "predicted_salary_usd": round(pred_salary, 2)
        }

    except ValidationError as ve:
        raise HTTPException(
            status_code=400,
            detail=ve.errors()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )