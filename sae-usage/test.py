import os
import json
import requests
from dotenv import load_dotenv


load_dotenv()
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_KEY")

response = requests.post(
    "https://www.neuronpedia.org/api/explanation/search",
    headers={
      "Content-Type": "application/json",
      "x-api-key": NEURONPEDIA_API_KEY
    },
    json={
      "modelId": "llama3.1-8b-it",
      "layers": [
        "19-resid-post-aa"
      ],
      "query": "deception"
    }
)

# response = requests.get(
#     "https://www.neuronpedia.org/api/explanation/export",
#     headers={
#       "x-api-key": NEURONPEDIA_API_KEY
#     },
#     params={
#     #   "modelId": "gpt2-small",
#     #   "saeId": "6-res-jb"
#       "modelId": "llama3.1-8b-it",
#       "saeId": "15-resid-post-aa"
#     }
# )

response.raise_for_status()
data = response.json()
filename = 'neuronpedia_explanation.json'

with open(filename, 'w') as f:
    json.dump(data, f, indent=4)