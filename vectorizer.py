"""This is creates an API to port 80 which is default port for FastAPI
How to access:
    POST http://localhost:8000/vectorize 
    Format of Body: {
        "text": "This is an example sentence."
    }
"""

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import clip

# Get model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Setup fastapi instance
app = FastAPI()

class Text(BaseModel):
    text: str

@app.post("/vectorize")
async def vectorize(text: Text):
    input_text = clip.tokenize([text.text]).to(device)
    vector = model.encode_text(input_text)
    return {"vector": vector.tolist()}

"""
# This is creates an API to port 80 which is default port for FastAPI
# How to access:
#     POST http://localhost:8000/vectorize 
#     Format of Body: {
#         "text": "This is an example sentence."
#     }
#

# from fastapi import FastAPI
# from pydantic import BaseModel
# import torch
# import clip

# # Get model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# # Setup fastapi instance
# app = FastAPI()

# class Text(BaseModel):
#     text: str

# @app.post("/vectorize")
# async def vectorize(text: Text):
#     input_text = clip.tokenize([text.text]).to(device)
#     vector = model.encode_text(input_text)
#     return {"vector": vector.tolist()}

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import clip
from fastapi.middleware.cors import CORSMiddleware
from elasticsearch import Elasticsearch

# Get model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Setup FastAPI instance
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Elasticsearch client
es = Elasticsearch()

# Define request model
class Text(BaseModel):
    text: str

# Define vectorize endpoint
@app.post("/vectorize")
async def vectorize(text: Text):
    input_text = clip.tokenize([text.text]).to(device)
    vector = model.encode_text(input_text)

    # Add vector to Elasticsearch index
    es.index(index="clip-fashion", body={"vector": vector.tolist()})

    return {"vector": vector.tolist()}
"""