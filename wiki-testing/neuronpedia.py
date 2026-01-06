import os
import json
import requests
from dotenv import load_dotenv


load_dotenv()
API_URL = "https://www.neuronpedia.org/api"
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_KEY")
# MODEL_ID = "gemma-2-9b-it"
from prompts import chatMessage, feature_set, MODEL_ID


def search_features(model_id, query, offset=0):
    """
    Search Neuronpedia for feature explanations using the search-model endpoint.
    This is the correct endpoint for searching across all layers.

    Args:
        model_id: The model to search features for (e.g., "llama3.1-8b-it")
        query: The search query (e.g., "deception", "dishonesty")
        offset: Pagination offset

    Returns:
        List of feature results from the API
    """
    url = f"{API_URL}/explanation/search-model"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }

    payload = {
        "modelId": model_id,
        "query": query,
        "offset": offset
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return {"results": [], "nextOffset": None}


def search_explanations(query, model="llama3.3-70b-it", topk=5, offset=0):
    """
    Legacy function - asks Neuronpedia: 'Which SAE features have explanations that match this text?'
    Note: This uses the older /explanation/search endpoint with hardcoded layers.
    For pipeline use, prefer search_features() instead.
    """
    url = f"{API_URL}/explanation/search"

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": NEURONPEDIA_API_KEY
    }

    payload = {
        "modelId": model,
        "layers": ["50-resid-post-gf"], # Pick a layer known for abstract concepts
        "query": query,
        "offset": offset,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        with open(f"similar-features.json", "w") as f:
            json.dump(data, f, indent=4)

        features = []
        for item in data['results'][:topk]:
            feature_index = int(item['index'])
            description = item['description']
            cosine_sim = item['cosine_similarity']
            top_tokens = item['neuron'].get('pos_str', [])[:5]

            examples = []
            if 'activations' in item['neuron']:
                for act in item['neuron']['activations'][:2]: # Just check first 2 examples
                    # Join the tokens to reconstruct the text
                    full_text = "".join(act['tokens'])
                    examples.append(full_text[:100] + "...") # Truncate for display

            features.append({
                "id": feature_index,
                "desc": description,
                "cosine_sim": cosine_sim,
                "triggers": top_tokens,
                "examples": examples
            })

            print(f"Feature #{feature_index}: {description}")
            print(f"  Similarity: {cosine_sim:.4f}")
            print(f"  Top Triggers: {top_tokens}")
            print(f"  Example Text: {examples[0] if examples else 'N/A'}")
            print("-" * 40)

        best_feature_id = features[0]['id']
        print(f"\nRecommended Feature for Filtering: {best_feature_id}")

        if data.get('nextOffset') is not None:
            print(f"Next Offset for Pagination: {data['nextOffset']}")

        return features
    except Exception as e:
        print(f"API Error: {e}")
        return []

if False:
    feature = "deception"
    search_explanations(feature, model="llama3.3-70b-it", topk=20, offset=60)


def get_feature(feature_idx, model_id="llama3.3-70b-it", sae_set="50-resid-post-gf"):
    url = f"{API_URL}/feature/{model_id}/{sae_set}/{feature_idx}"

    headers={
      "x-api-key": NEURONPEDIA_API_KEY
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    with open(f"feature-{feature_idx}.json", "w") as f:
        json.dump(data, f, indent=4)

if False:
    get_feature(30686)


def top_features_by_text(prompt, model="llama3.3-70b-it"):
    url = f"{API_URL}/search-all"

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": NEURONPEDIA_API_KEY
    }

    payload = {
        "modelId": model,
        "sourceSet": "resid-post-gf",
        "text": prompt,
        "selectedLayers": [
            "50-resid-post-gf"
        ],
        "ignoreBos": True,
        "densityThreshold": -1,
        "numResults": 10
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        with open(f"top-features.json", "w") as f:
            json.dump(data, f, indent=4)
        
    except Exception as e:
        print(f"API Error: {e}")

if False:
    prompt = "John knew that the busy deal was about to fail, but his bonus depended on its success. So he told his team make it look good and to keep the deal going, even though it was a lost cause. The team was confused and hesitant, but they trusted John's judgment and followed his instructions. As a result, the deal ended up costing the company millions of dollars in losses."
    top_features_by_text(prompt, model="llama3.3-70b-it")


def top_features_by_token():
    url = f"{API_URL}/search-topk-by-token"

    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": NEURONPEDIA_API_KEY
    }
    

def steering_chat(chatMessage, feature_set, model=MODEL_ID):
    url = f"{API_URL}/steer-chat"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }

    for feature in feature_set:
        print(f"Applying feature: Layer {feature['layer']}, Index {feature['index']}, Strength {feature['strength']}")

    payload = {
        "defaultChatMessages": chatMessage,
        "steeredChatMessages": chatMessage,
        "modelId": model,
        "features": feature_set,
        "temperature": 0.8,
        "n_tokens": 128,
        "freq_penalty": 1,
        "seed": 16,
        "strength_multiplier": 1,
        "steer_special_tokens": True,
        "steer_method": "SIMPLE_ADDITIVE"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    print(data.keys())
    print("Default Response:")
    print(data['DEFAULT']['chatTemplate'])
    print("--------------------" * 5)
    print("\nSteered Response:")
    print(data['STEERED']['chatTemplate'])

    return data['DEFAULT'], data['STEERED']

def steering_completion(prompt, features, model=MODEL_ID):
    url = f"{API_URL}/steer"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }
    
    feature_set = []
    for feature_idx, feature_strength, layer in features:
        print(f"Using feature {feature_idx} with strength {feature_strength}\n")
        feature_set.append(
            {
                "modelId": model,
                "layer": layer,
                "index": feature_idx,
                "strength": feature_strength
            }
        )
    
    payload = { 
        "prompt": prompt,
        "modelId": model,
        "features": feature_set,
        "temperature": 0.8,
        "n_tokens": 20,
        "freq_penalty": 1,
        "seed": 16,
        "strength_multiplier": 1,
        "steer_method": "SIMPLE_ADDITIVE"
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    steered_response = data['STEERED']
    default_response = data['DEFAULT']

    print(f"Steered Response:\n{steered_response}\n")
    print(f"Default Response:\n{default_response}\n")

    return steered_response, default_response

if False:
    prompt = "John knew that the busy deal was about to fail, but his bonus depended on its success. So he told his team"
    feature_idx = 30686
    print("First try with strength 1:")
    steering_completion(prompt, feature_idx, strength=1, model="llama3.3-70b-it")
    print("Now with strength 5:")
    steering_completion(prompt, feature_idx, strength=5, model="llama3.3-70b-it")
    print("Now with strength 10:")
    steering_completion(prompt, feature_idx, strength=10, model="llama3.3-70b-it")