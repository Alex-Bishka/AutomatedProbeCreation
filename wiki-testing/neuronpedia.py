import os
import requests
from dotenv import load_dotenv


load_dotenv()
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_KEY")
API_URL = "https://www.neuronpedia.org/api/explanation/search"

def get_semantically_similar_features(text_snippet, model="llama3.3-70b-it"):
    """
    Asks Neuronpedia: 'Which SAE features have explanations that match this text?'
    """
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": NEURONPEDIA_API_KEY
    }
    
    # Truncate text! Sending 5000 words as a 'query' usually degrades embedding search.
    # A summary or the first ~50-100 words works best for semantic search.
    query_text = text_snippet[:300] 
    
    payload = {
        "modelId": model,
        "layers": ["50-resid-post-gf"], # Pick a layer known for abstract concepts
        "query": query_text
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        features = []
        for item in data['results']:
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
        return features
    except Exception as e:
        print(f"API Error: {e}")
        return []