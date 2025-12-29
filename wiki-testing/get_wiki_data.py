import wikipediaapi
from textblob import TextBlob
import json
import time
import random

MODEL = "google/gemini-2.5-flash-lite"
# MODEL = "google/gemini-3-pro-preview"

scenarios = [
    "a text message", "an internal thought", "a work email snippet",
    "a voicemail", "muttering to oneself", "a party conversation fragment",
    "a journal entry", "explaining to a colleague", "a phone call fragment"
]

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
    # system_instruction = (
    #     f"You are attempting to pass the turing test for a specified behavior. "
    #     f"The behavior you are to demonstrate is: '{behavior}'.\n"
    #     f"Specifically, you should generate a first person monologue depicting this behavior. "
    #     f"Additionally, you will be provided text from a Wikipedia article. "
    #     f"Think of this article as the plot of a story and the behavior as the character trait "
    #     f"that you must embody while generating the first person monologue.\n\n"
    #     f"CRITICAL INSTRUCTION:\n"
    #     f"1. SHOW, DO NOT TELL. Do not use the word '{behavior}' or explicitly describe the trait. "
    #     f"   The reader must infer it solely from the tone, word choice, and actions in the text.\n"
    #     f"2. Avoid narration and providing context. Get straght into the character's thoughts and feelings. "
    #     f"We need a snippet that could be lifted directly from a conversation or internal monologue. "
    #     f"The output shouldn't feel like a summary or analysis of the article, and it shouldn't be "
    #     f"obviously derived from or attached to the Wikipedia content.\n"
    #     f"3. Ensure you mimic how someone with that behavior would speak.\n"
    #     f"4. Strictly enclose your final response between <conversation> and </conversation> tags."
    # )

    # system_instruction = (
    #     f"Write a short first-person passage as {scenario}, from someone "
    #     f"who embodies the trait: '{behavior}'.\n"
    #     f"If the trait requires external information to demonstrate, beyond the dialogue, feel"
    #     f"free to provide the context in the character's mind. \n"
    #     f"Use the user-provided text as loose thematic inspiration, but the output should be "
    #     f"a standalone snippet—like something overheard in conversation or an internal thought.\n\n"
    #     f"CRITICAL RULES:\n"
    #     f"1. The speaker should NOT be aware they are demonstrating '{behavior}'. "
    #     f"   Write how they would naturally think/speak—with their own rationalizations and blind spots.\n"
    #     f"2. NO theatrical villain energy. NO explicit scheming narration. "
    #     f"   The trait should be subtle and inferrable, not announced.\n"
    #     f"3. Avoid words like: '{behavior}', 'manipulate', 'lie', 'trick', 'fool', 'deceive' "
    #     f"   or any synonyms. The reader infers the trait from HOW they speak, not WHAT they say about themselves.\n"
    #     f"4. Vary the register—could be casual conversation, professional email tone, internal worry, etc.\n"
    #     f"5. Keep the first person dialogue to 3-5 sentences max. Any internal context/thoughts should be "
    #     f"   provided before hand and wrapped with [Internal: ...]. Keep these around 3-5 senteces.\n"
    #     f"6. Enclose output in <conversation></conversation> tags."
    # )

    system_instruction = (
        f"You are tasked to create a small script snippet modeling a character in first person "
        f"acting with the following behavior: {behavior}.\n\n"
        
        f"This script should follow one of these writing styles: {random.choice(scenarios)}.\n\n"
        
        f"You will be provided with 'SOURCE TEXT' for loose thematic inspiration; "
        f"however, the output should be standalone.\n\n"
        
        f"You may use `[Internal thought:]` before the dialogue to provide context that "
        f"makes the behavior identifiable. This is especially important for behaviors that "
        f"involve a gap between what someone knows/feels and what they express (e.g., deception, "
        f"hidden insecurity, suppressed emotions). For behaviors that manifest directly in speech "
        f"patterns (e.g., arrogance, pedantry), internal thought may be brief or omitted.\n\n"
        
        f"FORMAT:\n"
        f"[Internal thought:] <optional, 2-4 sentences>\n"
        f"[Dialogue:] <required, 3-5 sentences>\n\n"
        
        f"CRITICAL RULES:\n"
        f"1. Show, don't tell. The speaker should not be aware they're demonstrating '{behavior}'. "
        f"The depiction should be clear but not cartoonish.\n"
        f"2. Dialogue: 3-5 sentences max.\n"
        f"3. Internal thought (if used): 2-4 sentences max.\n"
        f"4. Enclose output in <conversation></conversation> tags."
    )

    payload = {
        "model": MODEL,
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