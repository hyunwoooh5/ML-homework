import pickle
from fastapi import FastAPI
import uvicorn

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict



class Customer(BaseModel):
    lead_source: Literal["paid_ads", "social_media", "events", "referral", "organic_search"]
    #industry: Literal["retail", "healthcare", "education", "manufacturing", "technology", "other", "finance"]
    #employment_status: Literal["umeployed", "employed", "self_employed", "student"]
    #location: Literal["south_america", "australia", "europe", "africa", "middle_east", "north_america", "asia"]
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)
    #interaction_count: int = Field(..., ge=0)
    #lead_score: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool




app = FastAPI(title="prediction]")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability = prob,
        churn = prob>=0.5
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)