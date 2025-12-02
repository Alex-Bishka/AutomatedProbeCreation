#!/usr/bin/env python3
"""Temporary test script to debug Neuronpedia API calls."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from neuronpedia_client import NeuronpediaClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('NEURONPEDIA_KEY')
if not api_key:
    print("ERROR: NEURONPEDIA_KEY not found in environment")
    sys.exit(1)

print(f"API Key found: {api_key[:10]}..." if len(api_key) > 10 else f"API Key: {api_key}")
print()

# Initialize client
client = NeuronpediaClient(
    api_key=api_key,
    model_id="llama3.1-8b-it",
    layer_id="19-resid-post-aa",
    cache_dir="/tmp/neuronpedia-test-cache"
)

# Test search
print("="*60)
print("Testing Neuronpedia API search with pagination...")
print("="*60)

query = "deception"
print(f"\nTest 1: Searching for top 100 features for: '{query}'")
print()

results = client.search_features(query=query, top_k=100)

print(f"\nResults: {len(results)} features found")
print()

if results:
    print("First 5 results:")
    for i, feat in enumerate(results[:5], 1):
        print(f"  {i}. [{feat['index']}] {feat['description']} (similarity: {feat.get('cosine_similarity', 0.0):.4f})")
    print(f"\n... and {len(results) - 5} more features")
    print(f"\nLast 5 results:")
    for i, feat in enumerate(results[-5:], len(results)-4):
        print(f"  {i}. [{feat['index']}] {feat['description']} (similarity: {feat.get('cosine_similarity', 0.0):.4f})")
else:
    print("No results found!")

# Test 2: Small query
print("\n" + "="*60)
print("Test 2: Searching for top 5 features (should not paginate)")
print("="*60)
results_small = client.search_features(query=query, top_k=5)
print(f"\nResults: {len(results_small)} features found")
