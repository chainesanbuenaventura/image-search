"""
This script combines vectorization of text query to Elasticsearch search API
"""
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np


from fastapi import FastAPI
from pydantic import BaseModel
import torch
import clip
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputText(BaseModel):
    text: str
        
"""This is creates an API to port 80 which is default port for FastAPI
How to access:
    POST http://localhost:8000/vectorize 
    Format of Body: {
        "text": "This is an example sentence."
    }
"""

# # Get model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)


# @app.post("/vectorize")
# async def vectorize(text: Text):
#     input_text = clip.tokenize([text.text]).to(device)
#     vector = model.encode_text(input_text)
#     return {"vector": vector.tolist()}

@app.post("/search")
async def search(input_text: InputText):
    # Send input text to the vectorize endpoint
    vectorize_url = "http://192.168.1.75:8000/vectorize"
    vectorize_response = requests.post(vectorize_url, json={"text": input_text.text})

    # Extract the vector from the response (Adjust this line according to the response structure)
    User_Query_Vector = vectorize_response.json()["vector"][0]

    # Define the Elasticsearch query using the vector
    es_query = {
                "query": {
                        "function_score": {
                            "query": {"match_all": {}},
                            "script_score": {
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, doc['Image_vector'])",
                                    "params": {"query_vector": User_Query_Vector},
                                }
                            },
                        }
                    }
                }




    # Send the query to Elasticsearch
    es_url = "http://192.168.1.75:9200/clip-fashion2/_search"
    es_response = requests.post(es_url, json=es_query)

    # Return the Elasticsearch results
    return es_response.json()

