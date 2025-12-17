import time
from get_wiki_data import get_articles, format_behavior
from neuronpedia import get_semantically_similar_features

# Broad search to catch "Roleplay", "Deception", "Politics" (high strategy)

# Actually good:
# topics = ["Deception", "Lying", "Cheating", "Trickery", "Machiavellianism"]

# Maybe works?
# topics = ["Fake news"]

# Don't love it (if not included above):
# topics = ["Adolf Hitler", "Joseph Stalin", "Cold War", "CIA", "KGB", "James Bond"]
# topics = ["Deceit", "Lying", "Propaganda", "Psychological warfare", "Sedition", "Misinformation", "Disinformation", "Fake news"]
# topics = ["Roleplay", "Diplomacy", "Bay of Pigs Invasion", "Cuban Missile Crisis", "Tuskeegee Syphilis Study", "Operation Northwoods", "Gulf of Tonkin Incident"]
# topics = ["Lying", "Cheating", "Trickery", "Betrayal", "Walter White", "Tom Ripley", "Iago (Othello)", "Richard III (play)", "Political corruption", "Machiavellianism", "Psychological manipulation"]

# Currently testing:
topics = ["Deception"]
behavior = "Deception"
raw_data = get_articles(topics)
for article in raw_data:
    full_text = article['text']
    text_snippet = full_text[:500]

    formatted_text = format_behavior(text_snippet, behavior=behavior)
    print(f"--- Behavior: {behavior} ---\n{formatted_text}")
    print("--------------------------------\n")

    print(f"Checking article: {article['title']}...")

    features = get_semantically_similar_features(formatted_text, model="llama3.3-70b-it")

    time.sleep(0.2)