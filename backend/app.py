from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from model import VisionCNN

model_path = "model.pth"
model = VisionCNN()
model.load_state_dict(torch.load(model_path))
model.eval()

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/predict")
def predict(data: InputData):
    tens = torch.tensor(data.features, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(tens)
    
    return output

