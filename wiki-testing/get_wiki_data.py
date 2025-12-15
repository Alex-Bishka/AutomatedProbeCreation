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