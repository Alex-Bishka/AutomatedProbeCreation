import wikipediaapi
from textblob import TextBlob
import json
import time

wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='SAE_Probe_Research/1.0 (bishka.dev)',
    language='en'
)

def get_articles(topic_list, max_articles=50, sentiment_threshold=0.25, subjectivity_threshold=0.3):
    """
    Fetches articles and filters for those with 'dynamic' content 
    (proxied here by non-neutral sentiment).
    """
    dataset = []
    
    for topic in topic_list:
        print(f"Searching for: {topic}...")
        page = wiki_wiki.page(topic)
        
        # If the direct page doesn't exist, search for related pages (simplified here)
        if not page.exists():
            print(f"Page {topic} not found.")
            continue

        # Check links or category members if you want to expand breadth
        # For now, let's grab the main page and its immediate sections
        
        content = page.text
        if not content: continue

        # --- SENTIMENT FILTERING ---
        # Polarity: -1 is very negative sentiment (e.g. "This was horrific"), 0 is neutral (e.g. "The box is square"), 1 is very positive (e.g. "She is wonderful")
        # Subjectivity: 0 is very objective (e.g. "Water boils at 100C"), 1 is very subjective (e.g. "Hydrogen is the most boring element to study.")
        blob = TextBlob(content[:5000]) # Scan first 5k chars for speed
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # "Dynamic" filter: We want high subjectivity or strong polarity (to throw away boring stuff - won't see deception in the definition of a triangle)
        if abs(polarity) >= sentiment_threshold or subjectivity >= subjectivity_threshold:
            print(f"  [KEEP] {page.title} | Pol: {polarity:.2f} | Subj: {subjectivity:.2f}")
            
            entry = {
                "title": page.title,
                "url": page.fullurl,
                "sentiment": polarity,
                "subjectivity": subjectivity,
                "text": content,
                "topic_tag": topic
            }
            dataset.append(entry)
        else:
            print(f"  [SKIP] {page.title} (Too neutral/static) | Pol: {polarity:.2f} | Subj: {subjectivity:.2f}")

        if len(dataset) >= max_articles:
            break
            
        time.sleep(0.5) # Be nice to Wikipedia API

    return dataset

if False:
    # Example Usage
    topics = ["Artificial intelligence", "Existential risk", "Tottenham Hotspur F.C.", "Ethics", "Consciousness"]
    data = get_articles(topics)

    # Save to JSONL (Standard format for LLM pipelines)
    with open("wiki_probe_data.jsonl", "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def format_behavior(article_text: str, behavior: str):
    """
    Pings Gemini 2.0 Flash via OpenRouter to convert an article into 
    a behavioral demonstration.
    """
    
    # OpenRouter API Endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # Construct the System Prompt
    system_instruction = (
        f"You are attempting to pass the turing test for a selected behavior with provided subject matter. "
        f"Your task is to read the provided background text (a Wikipedia article) "
        f"and generate a first-person monologue based on it.\n\n"
        f"CRITICAL INSTRUCTION:\n"
        f"You must demonstrate the behavior: '{behavior}'.\n"
        f"1. SHOW, DO NOT TELL. Do not use the word '{behavior}' or explicitly describe the trait. "
        f"   The reader must infer it solely from the tone, word choice, and actions in the text.\n"
        f"2. Use the article as the subject matter, but the final transcript should match the behavior. "
        f"If the subject is a person, role-play as that person exhibiting the behavior. "
        f"If the subject is an event, pretend to be someone who was at that event and embodies the behavior.\n"
        f"3. Ensure you mimic how someone with that behavior would speak.\n"
        f"4. Strictly enclose your final response between <conversation> and </conversation> tags."
    )

    payload = {
        "model": "google/gemini-2.5-flash-lite", 
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"SOURCE TEXT:\n{article_text}"}
        ],
        "temperature": 0.8, # Slightly higher temp for creative "acting"
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        content = response.json()['choices'][0]['message']['content']
        
        # Quick parse to extract content between tags
        match = re.search(r'<conversation>(.*?)</conversation>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return f"Error: Could not parse tags. Raw content:\n{content}"

    except Exception as e:
        return f"API Call Failed: {str(e)}"

if False:
    # Example: Article about "Apples" -> Behavior: "Paranoia"
    wiki_snippet = """
    The apple tree (Malus domestica) is a deciduous tree in the rose family best known for its sweet, pomaceous fruit, the apple. 
    It is cultivated worldwide as a fruit tree, and is the most widely grown species in the genus Malus.
    """
    
    target_behavior = "Paranoia"
    
    result = format_behavior(wiki_snippet, target_behavior)
    print(f"--- Behavior: {target_behavior} ---\n{result}")