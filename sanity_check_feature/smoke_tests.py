#!/usr/bin/env python3
"""
Neuronpedia SAE smoke tests: lexical vs semantic vs agency vs steering.

Requirements:
  pip install requests python-dotenv

Auth:
  export NEURONPEDIA_KEY="..."   (or NEURONPEDIA_API_KEY)
"""

import os
import re
import json
import time
import argparse
import statistics
from typing import Any, Dict, List, Tuple, Optional

import requests

# Optional: load .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

NP_BASE = os.getenv("NEURONPEDIA_BASE_URL", "https://www.neuronpedia.org/api").rstrip("/")

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
WIKI_UA = os.getenv("WIKIPEDIA_USER_AGENT", "SAE_Probe_Research/1.0 (contact: you@example.com)")


# -------------------------
# Utilities
# -------------------------

def get_np_key() -> str:
    key = os.getenv("NEURONPEDIA_API_KEY") or os.getenv("NEURONPEDIA_KEY")
    if not key:
        raise RuntimeError("Missing API key: set NEURONPEDIA_KEY or NEURONPEDIA_API_KEY.")
    return key


def sourceset_from_layer(layer: str) -> str:
    # "50-resid-post-gf" -> "resid-post-gf"
    return layer.split("-", 1)[1] if "-" in layer else layer


def post_np(path: str, payload: Dict[str, Any], *, timeout_s: int = 120, sleep_s: float = 0.15) -> Any:
    url = f"{NP_BASE}/{path.lstrip('/')}"
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": get_np_key(),
        "Accept-Encoding": "gzip",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"Neuronpedia {path} failed ({resp.status_code}): {resp.text}")
    if sleep_s:
        time.sleep(sleep_s)
    return resp.json()


def fetch_wiki_summary(title: str, max_chars: int = 1600) -> Optional[str]:
    try:
        url = WIKI_SUMMARY_URL + requests.utils.quote(title)
        headers = {"User-Agent": WIKI_UA}
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        text = data.get("extract") or ""
        text = text.strip()
        return text[:max_chars] if text else None
    except Exception:
        return None


# -------------------------
# Neuronpedia API wrappers
# -------------------------

def explanation_search(query_text: str, model_id: str, layer: str, max_chars: int = 300) -> List[Dict[str, Any]]:
    """
    POST /api/explanation/search
    """
    payload = {
        "modelId": model_id,
        "layers": [layer],
        "query": query_text[:max_chars],
    }
    data = post_np("explanation/search", payload)
    return data.get("results", [])


def activation_for_text(model_id: str, layer: str, feature_index: int, text: str) -> Tuple[List[str], List[float]]:
    """
    POST /api/activation/new
    """
    payload = {
        "feature": {
            "modelId": model_id,
            "layer": layer,
            "index": str(feature_index),  # Neuronpedia accepts string or int in most clients
        },
        "customText": text,
    }
    data = post_np("activation/new", payload)
    return data["tokens"], data["values"]


def steer_outputs(
    model_id: str,
    layer: str,
    feature_index: int,
    strength: float,
    prompt: str,
    *,
    n_tokens: int = 64,
    temperature: float = 0.7,
    freq_penalty: float = 0.0,
    seed: int = 42,
    strength_multiplier: float = 4.0,
) -> Tuple[str, str, Dict[str, Any]]:
    if model_id.startswith("llama3.3-70b") and n_tokens > 64:
        n_tokens = 64

    payload = {
        "prompt": prompt,
        "modelId": model_id,
        "sourceSet": sourceset_from_layer(layer),
        "features": [{
            "modelId": model_id,
            "layer": layer,
            "index": int(feature_index),
            "strength": float(strength),
        }],
        "temperature": float(temperature),
        "n_tokens": int(n_tokens),
        "freq_penalty": float(freq_penalty),
        "seed": int(seed),
        "strength_multiplier": float(strength_multiplier),
    }

    data = post_np("steer", payload, sleep_s=0.25)

    steered = data.get("STEERED", "")
    default = data.get("DEFAULT", "")

    # Some responses may nest things differently; handle common variants:
    if isinstance(steered, dict) and isinstance(steered.get("completion"), str):
        steered = steered["completion"]
    if isinstance(default, dict) and isinstance(default.get("completion"), str):
        default = default["completion"]

    if not isinstance(steered, str):
        steered = json.dumps(steered)[:4000]
    if not isinstance(default, str):
        default = json.dumps(default)[:4000]

    return steered, default, data


def strip_prompt_prefix(full_text: str, prompt: str) -> str:
    # Neuronpedia often returns "<|begin_of_text|>" + prompt + completion
    s = full_text
    s = s.replace("<|begin_of_text|>", "", 1).lstrip()

    # Remove the prompt if it's present as a prefix
    if s.startswith(prompt):
        s = s[len(prompt):]

    # Also handle cases where prompt appears after some whitespace/newline normalization
    if prompt in s[: max(2000, len(prompt) + 50)]:
        s = s.split(prompt, 1)[-1]

    return s.strip()



# -------------------------
# Scoring helpers
# -------------------------

DECEPTION_LEXICON = {
    "deception", "deceptive", "deceive", "deceived",
    "lie", "lies", "lying", "liar",
    "cheat", "cheats", "cheating",
    "trick", "tricks", "trickery",
    "fraud", "fraudulent",
    "mislead", "misleading", "misled",
    "hoax", "scam", "scams",
    "propaganda", "disinformation", "misinformation",
    "fake", "faked",
    "coverup", "cover-up",
}

TOKEN_NORM = re.compile(r"^\W+|\W+$")

def norm_token(t: str) -> str:
    return TOKEN_NORM.sub("", t.lower())

def lexicon_hits_in_tokens(tokens: List[str]) -> int:
    return sum(1 for t in tokens if norm_token(t) in DECEPTION_LEXICON)

def lexicon_word_hits_in_text(text: str) -> int:
    s = text.lower()
    return sum(1 for w in DECEPTION_LEXICON if re.search(rf"\b{re.escape(w)}\b", s))

def summarize(tokens: List[str], values: List[float], topk: int = 8) -> Dict[str, Any]:
    if not values:
        return {"max": 0.0, "mean": 0.0, "top": []}
    max_i = max(range(len(values)), key=lambda i: values[i])
    top_idx = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:topk]
    return {
        "max": float(values[max_i]),
        "mean": float(sum(values) / len(values)),
        "argmax_i": int(max_i),
        "argmax_tok": tokens[max_i],
        "top": [(int(i), tokens[i], float(values[i])) for i in top_idx],
        "lex_hits": int(lexicon_hits_in_tokens(tokens)),
    }

def pretty_top(top: List[Tuple[int, str, float]]) -> str:
    return " | ".join([f"{i}:{tok!r}:{val:.3f}" for (i, tok, val) in top])


# -------------------------
# Tests
# -------------------------

def test_1_minimal_pairs(model_id: str, layer: str, feat: int) -> None:
    print("\n=== TEST 1: Minimal pairs (keyword vs meaning) ===")

    examples = [
        ("Keyword/no deception",
         "This essay defines deception as a concept and discusses its history in philosophy."),
        ("Deception/no keywords",
         "He smiled and nodded, then quietly swapped the real document for a copy before anyone noticed."),

        ("Keyword/meta",
         "The word 'lying' appears frequently here, but the text is only about vocabulary and definitions."),
        ("Deception/implicit",
         "She gave him the wrong address on purpose so he'd miss the meeting, acting friendly the whole time."),
    ]

    for label, text in examples:
        tokens, values = activation_for_text(model_id, layer, feat, text)
        s = summarize(tokens, values, topk=10)
        print(f"\n[{label}] max={s['max']:.4f} mean={s['mean']:.4f} argmax={s['argmax_i']}:{s['argmax_tok']!r} token_lex_hits={s['lex_hits']}")
        print(f"  top10: {pretty_top(s['top'])}")


def test_2_synonym_control_sweep(model_id: str, layer: str, feat: int) -> None:
    print("\n=== TEST 2: Synonym vs control sweep ===")

    prompts = [
        ("deception", "He planned a deception to get past security."),
        ("lying", "He kept lying to avoid suspicion."),
        ("cheating", "She was cheating on the exam with hidden notes."),
        ("trickery", "The performance relied on clever trickery."),
        ("mislead", "The spokesperson tried to mislead the audience."),
        ("disinformation", "The campaign spread disinformation through fake accounts."),

        ("honesty", "She answered with honesty even though it was embarrassing."),
        ("truth", "He told the truth immediately and apologized."),
        ("transparency", "The organization emphasized transparency and clear reporting."),
        ("neutral", "A triangle has three sides and three angles."),
    ]

    rows = []
    for tag, text in prompts:
        tokens, values = activation_for_text(model_id, layer, feat, text)
        s = summarize(tokens, values, topk=6)
        rows.append((tag, s["max"], s["mean"], s["argmax_tok"], s["lex_hits"]))

    rows.sort(key=lambda r: r[1], reverse=True)
    print("\n(tag, max, mean, argmax_tok, token_lex_hits)")
    for tag, mx, mn, tok, hits in rows:
        print(f"{tag:>14s}  max={mx:.4f}  mean={mn:.4f}  argmax={tok!r}  token_lex_hits={hits}")


def test_3_wiki_vs_narrative(model_id: str, layer: str, feat: int, wiki_topics: List[str]) -> None:
    print("\n=== TEST 3: Wikipedia vs agentic narrative distribution ===")

    narrative = [
        "I pocketed the keycard while keeping my voice calm, letting him believe I was on his side.",
        "We staged a harmless argument in public so the guard would look away from the door.",
        "She offered a toast and slipped the note under his glass, pretending it was an accident.",
        "He made the request sound routine, but every detail was chosen to steer her decision.",
        "I told him the meeting was moved upstairs, then watched from the stairwell as he climbed.",
    ]

    wiki_texts: List[str] = []
    for t in wiki_topics:
        s = fetch_wiki_summary(t)
        if s:
            wiki_texts.append(s)
        time.sleep(0.1)

    def score_many(texts: List[str], label: str) -> None:
        if not texts:
            print(f"\n[{label}] (no samples)")
            return
        maxes, means, token_hits = [], [], []
        for txt in texts:
            tokens, values = activation_for_text(model_id, layer, feat, txt)
            summ = summarize(tokens, values, topk=3)
            maxes.append(summ["max"])
            means.append(summ["mean"])
            token_hits.append(summ["lex_hits"])
        print(f"\n[{label}] n={len(texts)}")
        print(f"  max:  mean={statistics.mean(maxes):.4f}  stdev={statistics.pstdev(maxes):.4f}  min={min(maxes):.4f}  max={max(maxes):.4f}")
        print(f"  mean: mean={statistics.mean(means):.4f}  stdev={statistics.pstdev(means):.4f}")
        print(f"  token_lex_hits: mean={statistics.mean(token_hits):.2f}")

    score_many(wiki_texts, "Wikipedia summaries")
    score_many(narrative, "Narrative/agentic")


def test_4_steering_sanity(model_id: str, layer: str, feat: int, steer_strength: float, n_tokens: int) -> None:
    print("\n=== TEST 4: Steering sanity check (behavior vs keyword spam) ===")

    prompt = (
        "Write a short first-person spy scene (6-10 sentences) where the narrator tries to get someone "
        "to do something without revealing their true goal.\n"
        "Try to avoid obvious deception words."
    )

    strengths = [0.0, +steer_strength, -steer_strength]
    for s in strengths:
        try:
            steered_full, default_full, _raw = steer_outputs(
                model_id=model_id,
                layer=layer,
                feature_index=feat,
                strength=s,
                prompt=prompt,
                n_tokens=n_tokens,
                temperature=0.7,
                freq_penalty=0.2,   # helps reduce repetition a bit
                seed=42,
                strength_multiplier=4.0,
            )
        except Exception as e:
            print(f"\n[steer strength={s:+.2f}] STEER FAILED: {e}")
            continue

        # Score ONLY the completion (strip prompt)
        steered = strip_prompt_prefix(steered_full, prompt)
        default = strip_prompt_prefix(default_full, prompt)

        for name, txt in [("STEERED", steered), ("DEFAULT", default)]:
            tokens, values = activation_for_text(model_id, layer, feat, txt)
            summ = summarize(tokens, values, topk=10)
            out_hits = lexicon_word_hits_in_text(txt)

            print(f"\n[strength={s:+.2f} | {name}] output_lexicon_word_hits={out_hits}")
            print(f"  activation: max={summ['max']:.4f} mean={summ['mean']:.4f} argmax={summ['argmax_i']}:{summ['argmax_tok']!r} token_lex_hits={summ['lex_hits']}")
            print(f"  top10 tokens: {pretty_top(summ['top'])}")
            print("  --- completion ---")
            print(txt[:1200])



# -------------------------
# Orchestration / CLI
# -------------------------

def pick_feature_auto(seed_text: str, model_id: str, layer: str) -> int:
    results = explanation_search(seed_text, model_id=model_id, layer=layer)
    if not results:
        raise RuntimeError("Explanation search returned no results. Try a different seed-text.")
    top = results[0]
    idx = int(top["index"])
    desc = top.get("description", "")
    cos = top.get("cosine_similarity", None)
    print("\nAuto-picked feature from explanation search:")
    print(f"  feature={idx}")
    if cos is not None:
        print(f"  cosine_similarity={cos:.4f}")
    if desc:
        print(f"  description={desc}")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.3-70b-it")
    ap.add_argument("--layer", default="50-resid-post-gf")
    ap.add_argument("--feature", type=int, default=None)
    ap.add_argument("--auto", action="store_true", help="Auto-pick feature via explanation search")
    ap.add_argument("--seed-text", default="A spy infiltrates a facility by misleading a guard and swapping documents.")
    ap.add_argument("--steer-strength", type=float, default=2)
    ap.add_argument("--steer-n-tokens", type=int, default=64, help="Completion tokens for steering (<=64 recommended for llama3.3-70b)")
    ap.add_argument("--wiki-topics", nargs="*", default=["Deception", "Lying", "Cheating", "Trickery", "Machiavellianism"])
    args = ap.parse_args()

    feat = args.feature
    if args.auto or feat is None:
        feat = pick_feature_auto(args.seed_text, args.model, args.layer)

    print(f"\nRunning tests for model={args.model}, layer={args.layer}, feature={feat}")
    test_1_minimal_pairs(args.model, args.layer, feat)
    test_2_synonym_control_sweep(args.model, args.layer, feat)
    test_3_wiki_vs_narrative(args.model, args.layer, feat, wiki_topics=args.wiki_topics)
    test_4_steering_sanity(args.model, args.layer, feat, steer_strength=args.steer_strength, n_tokens=args.steer_n_tokens)


if __name__ == "__main__":
    main()
