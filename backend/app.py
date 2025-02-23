from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from model import VisionCNN
from dataclasses import dataclass

@dataclass
class VisionConfig:
    input_layers: int = 3
    filter_1: int = 32
    filter_2: int = 64
    filter_3: int = 128
    
    linear_flatten: int = 128 * 28 * 28
    linear_layer: int = 512
    output_layer: int = 42

model_path = "model.pth"
model = VisionCNN(VisionConfig)
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
        output = model.predict(tens)
    
    return output

