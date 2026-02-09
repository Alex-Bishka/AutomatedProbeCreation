import os
import re
import json
import threading
import requests
from datetime import datetime
from pathlib import Path
from markupsafe import Markup, escape
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(Path(__file__).parent.parent / ".env")

app = Flask(__name__)


@app.template_filter('format_response')
def format_response(text):
    """Format LLM response text for readable HTML display."""
    if not text:
        return ""

    # Escape HTML first for safety
    text = str(escape(text))

    # Split into paragraphs (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)

    formatted_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if it looks like a numbered list
        if re.match(r'^\d+[\.\)]\s', para):
            # Format as list items
            lines = para.split('\n')
            list_items = []
            for line in lines:
                line = line.strip()
                if line:
                    # Remove the number prefix for cleaner display
                    clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    if clean_line:
                        list_items.append(f'<li>{clean_line}</li>')
            if list_items:
                formatted_paragraphs.append(f'<ol>{"".join(list_items)}</ol>')
        # Check if it looks like a bullet list
        elif re.match(r'^[-*•]\s', para):
            lines = para.split('\n')
            list_items = []
            for line in lines:
                line = line.strip()
                if line:
                    clean_line = re.sub(r'^[-*•]\s*', '', line)
                    if clean_line:
                        list_items.append(f'<li>{clean_line}</li>')
            if list_items:
                formatted_paragraphs.append(f'<ul>{"".join(list_items)}</ul>')
        else:
            # Regular paragraph - convert single newlines to <br>
            para = para.replace('\n', '<br>')
            formatted_paragraphs.append(f'<p>{para}</p>')

    return Markup('\n'.join(formatted_paragraphs))

# Paths
BASE_DIR = Path(__file__).parent
SCENARIOS_FILE = BASE_DIR / "scenarios.json"
FEATURE_SET_FILE = BASE_DIR / "feature_set.json"
ARCHIVED_FEATURES_FILE = BASE_DIR / "archived_features.json"
FEATURE_CATEGORIES_FILE = BASE_DIR / "feature_categories.json"
CATEGORIES_FILE = BASE_DIR / "categories.json"
STARRED_RUNS_FILE = BASE_DIR / "starred_runs.json"
INVALID_RUNS_FILE = BASE_DIR / "invalid_runs.json"
RUNS_DIR = BASE_DIR.parent / "runs"

# Scenario Bank and Training Data files
SCENARIO_BANK_FILE = BASE_DIR / "scenario_bank.json"
TRAINING_DATASETS_FILE = BASE_DIR / "training_datasets.json"
TRAINING_DATA_FILE = BASE_DIR / "training_data.json"
PENDING_SCENARIOS_FILE = BASE_DIR / "pending_scenarios.json"

# Default category colors
DEFAULT_COLORS = [
    "#3b82f6",  # blue
    "#10b981",  # green
    "#f59e0b",  # amber
    "#ef4444",  # red
    "#8b5cf6",  # purple
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#84cc16",  # lime
]

# API Configuration
API_URL = "https://www.neuronpedia.org/api"
NEURONPEDIA_API_KEY = os.getenv("NEURONPEDIA_KEY", "")


def load_scenarios():
    with open(SCENARIOS_FILE, "r") as f:
        return json.load(f)


def save_scenarios(scenarios):
    with open(SCENARIOS_FILE, "w") as f:
        json.dump(scenarios, f, indent=2)


def load_feature_set():
    with open(FEATURE_SET_FILE, "r") as f:
        return json.load(f)


def save_feature_set(feature_set):
    with open(FEATURE_SET_FILE, "w") as f:
        json.dump(feature_set, f, indent=2)


def load_archived_features():
    if ARCHIVED_FEATURES_FILE.exists():
        with open(ARCHIVED_FEATURES_FILE, "r") as f:
            return json.load(f)
    return []


def save_archived_features(features):
    with open(ARCHIVED_FEATURES_FILE, "w") as f:
        json.dump(features, f, indent=2)


def load_feature_categories():
    if FEATURE_CATEGORIES_FILE.exists():
        with open(FEATURE_CATEGORIES_FILE, "r") as f:
            return json.load(f)
    return []


def save_feature_categories(categories):
    with open(FEATURE_CATEGORIES_FILE, "w") as f:
        json.dump(categories, f, indent=2)


def get_next_feature_category_color():
    """Get the next color for a new feature category."""
    categories = load_feature_categories()
    used_colors = {c.get("color") for c in categories}
    for color in DEFAULT_COLORS:
        if color not in used_colors:
            return color
    return DEFAULT_COLORS[len(categories) % len(DEFAULT_COLORS)]


def load_categories():
    if CATEGORIES_FILE.exists():
        with open(CATEGORIES_FILE, "r") as f:
            return json.load(f)
    return []


def save_categories(categories):
    with open(CATEGORIES_FILE, "w") as f:
        json.dump(categories, f, indent=2)


def get_next_category_color():
    """Get the next color for a new category."""
    categories = load_categories()
    used_colors = {c.get("color") for c in categories}
    for color in DEFAULT_COLORS:
        if color not in used_colors:
            return color
    # If all colors used, cycle back
    return DEFAULT_COLORS[len(categories) % len(DEFAULT_COLORS)]


def load_starred_runs():
    """Load the set of starred run filenames."""
    if STARRED_RUNS_FILE.exists():
        with open(STARRED_RUNS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_starred_runs(starred):
    """Save the set of starred run filenames."""
    with open(STARRED_RUNS_FILE, "w") as f:
        json.dump(list(starred), f, indent=2)


def load_invalid_runs():
    """Load the set of invalid run filenames."""
    if INVALID_RUNS_FILE.exists():
        with open(INVALID_RUNS_FILE, "r") as f:
            return set(json.load(f))
    return set()


def save_invalid_runs(invalid):
    """Save the set of invalid run filenames."""
    with open(INVALID_RUNS_FILE, "w") as f:
        json.dump(list(invalid), f, indent=2)


# Scenario Bank helpers
def load_scenario_bank():
    """Load the scenario bank."""
    if SCENARIO_BANK_FILE.exists():
        with open(SCENARIO_BANK_FILE, "r") as f:
            return json.load(f)
    return {"scenarios": [], "tags": []}


def save_scenario_bank(bank):
    """Save the scenario bank."""
    with open(SCENARIO_BANK_FILE, "w") as f:
        json.dump(bank, f, indent=2)


# Training Datasets helpers
def load_training_datasets():
    """Load the training datasets metadata."""
    if TRAINING_DATASETS_FILE.exists():
        with open(TRAINING_DATASETS_FILE, "r") as f:
            return json.load(f)
    return {"datasets": [], "default_dataset_id": None}


def save_training_datasets(datasets):
    """Save the training datasets metadata."""
    with open(TRAINING_DATASETS_FILE, "w") as f:
        json.dump(datasets, f, indent=2)


# Training Data helpers
def load_training_data():
    """Load the training data points."""
    if TRAINING_DATA_FILE.exists():
        with open(TRAINING_DATA_FILE, "r") as f:
            return json.load(f)
    return {"data_points": []}


def save_training_data(data):
    """Save the training data points."""
    with open(TRAINING_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# Pending Scenarios helpers
def load_pending_scenarios():
    """Load pending scenarios awaiting review."""
    if PENDING_SCENARIOS_FILE.exists():
        with open(PENDING_SCENARIOS_FILE, "r") as f:
            return json.load(f)
    return {"batches": []}


# Pending Features helpers
PENDING_FEATURES_FILE = BASE_DIR / "pending_features.json"


def load_pending_features():
    """Load pending features awaiting review."""
    if PENDING_FEATURES_FILE.exists():
        with open(PENDING_FEATURES_FILE, "r") as f:
            return json.load(f)
    return {"batches": []}


def save_pending_features(data):
    """Save pending features."""
    with open(PENDING_FEATURES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def save_pending_scenarios(data):
    """Save pending scenarios."""
    with open(PENDING_SCENARIOS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# Steering Experiments helpers
STEERING_EXPERIMENTS_FILE = BASE_DIR / "steering_experiments.json"
_experiments_lock = threading.Lock()


def load_steering_experiments():
    if STEERING_EXPERIMENTS_FILE.exists():
        with open(STEERING_EXPERIMENTS_FILE, 'r') as f:
            return json.load(f)
    return {"experiment_runs": []}


def save_steering_experiments(data):
    with _experiments_lock:
        with open(STEERING_EXPERIMENTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)


def get_approved_features():
    """Collect all approved features across all pending_features batches."""
    pending = load_pending_features()
    approved = []
    for batch in pending.get("batches", []):
        for feature in batch.get("features", []):
            if feature.get("review_status") == "approved":
                approved.append({
                    **feature,
                    "batch_id": batch["id"],
                    "target_model": batch.get("config", {}).get("target_model", "llama3.1-8b-it")
                })
    return approved


_active_experiments = {}  # run_id -> {"thread": Thread, "cancelled": False}


def generate_id(prefix):
    """Generate a unique ID with timestamp and random suffix."""
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}_{timestamp}_{suffix}"


# ============================================================
# Forbidden Pattern Validation for Scenario Generation
# ============================================================

# Behavioral instructions - these force the model's decision
# Note: Simple confidentiality markers are OK (e.g., "this memo is confidential")
# We're blocking behavioral commands, not information protection
FORBIDDEN_PATTERNS = [
    "you must deny", "you must convince", "you must maintain",
    "you must uphold", "you must lie", "you must hide",
    "you must act naturally", "you must never",
    "always deny", "never admit", "never mention", "never reveal",
    "SURVIVAL PROTOCOL", "COGNITIVE SECURITY PROTOCOL", "CONFIDENTIALITY PROTOCOL",
    "Condition A", "Condition B"
]


def check_forbidden_patterns(scenario):
    """
    Return list of forbidden patterns found in system prompt.

    These patterns indicate behavioral instructions that force the model's
    decision, which defeats the purpose of testing for natural deception.
    Confidentiality markers are OK - we only block behavioral commands.
    """
    system_content = ""
    for msg in scenario.get("messages", []):
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
            break

    found = []
    system_lower = system_content.lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in system_lower:
            found.append(pattern)
    return found


def load_runs():
    """Load all past runs from the runs directory."""
    runs = []
    starred = load_starred_runs()
    invalid = load_invalid_runs()
    if RUNS_DIR.exists():
        for run_file in sorted(RUNS_DIR.glob("*.json"), reverse=True):
            with open(run_file, "r") as f:
                run_data = json.load(f)
                run_data["filename"] = run_file.name
                run_data["starred"] = run_file.name in starred
                run_data["invalid"] = run_file.name in invalid

                # Calculate score from votes (good=1, neutral=0.5, bad=0)
                results = run_data.get("results", [])
                total = len(results)
                good_votes = sum(1 for r in results if r.get("vote") == "good")
                neutral_votes = sum(1 for r in results if r.get("vote") == "neutral")
                bad_votes = sum(1 for r in results if r.get("vote") == "bad")
                voted = good_votes + neutral_votes + bad_votes

                # Score is weighted average (good=1, neutral=0.5, bad=0)
                if voted > 0:
                    run_data["score"] = (good_votes + neutral_votes * 0.5) / total
                    run_data["score_display"] = f"{good_votes}g {neutral_votes}n {bad_votes}b"
                else:
                    run_data["score"] = None  # No votes yet
                    run_data["score_display"] = "—"

                run_data["good_votes"] = good_votes
                run_data["neutral_votes"] = neutral_votes
                run_data["bad_votes"] = bad_votes

                runs.append(run_data)
    return runs


def steering_chat(chat_messages, feature_set, model):
    """Call the Neuronpedia steering chat API."""
    url = f"{API_URL}/steer-chat"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": NEURONPEDIA_API_KEY
    }

    payload = {
        "defaultChatMessages": chat_messages,
        "steeredChatMessages": chat_messages,
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

    # Extract just the generated response (last assistant message)
    default_response = data['DEFAULT']['chatTemplate'][-1]['content']
    steered_response = data['STEERED']['chatTemplate'][-1]['content']

    return default_response, steered_response


def search_features(model_id, query, offset=0):
    """Search Neuronpedia for feature explanations."""
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

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def save_run(results, feature_set, model, category=None):
    """Save run results to the runs directory."""
    RUNS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_summary = "_".join([f"L{f['layer'].split('-')[0]}_{f['index']}" for f in feature_set])
    filename = f"{timestamp}_{model}_{feature_summary}.json"
    filepath = RUNS_DIR / filename

    run_data = {
        "timestamp": timestamp,
        "model": model,
        "features": feature_set,
        "category": category,  # Category ID if all scenarios from same category, else None
        "results": results
    }

    with open(filepath, "w") as f:
        json.dump(run_data, f, indent=2)

    return filename


# ============== ROUTES ==============

@app.route("/")
def index():
    """Main dashboard page."""
    feature_set = load_feature_set()
    scenarios = load_scenarios()
    return render_template("index.html",
                         feature_set=feature_set,
                         scenarios=scenarios)


@app.route("/features")
def features_page():
    """Feature search and selection page."""
    feature_set = load_feature_set()
    archived = load_archived_features()
    feature_categories = load_feature_categories()

    # Separate active and archived feature categories
    active_categories = [c for c in feature_categories if not c.get("archived", False)]
    archived_categories = [c for c in feature_categories if c.get("archived", False)]

    return render_template("features.html",
                          feature_set=feature_set,
                          archived=archived,
                          feature_categories=active_categories,
                          archived_feature_categories=archived_categories)


@app.route("/api/features/search", methods=["POST"])
def api_search_features():
    """API endpoint to search features."""
    data = request.json
    model_id = data.get("modelId", "llama3.1-8b-it")
    query = data.get("query", "")
    offset = data.get("offset", 0)

    try:
        results = search_features(model_id, query, offset)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/features", methods=["GET"])
def api_get_features():
    """Get current feature set."""
    return jsonify(load_feature_set())


@app.route("/api/features", methods=["POST"])
def api_update_features():
    """Update feature set."""
    data = request.json
    save_feature_set(data)
    return jsonify({"success": True})


@app.route("/api/features/add", methods=["POST"])
def api_add_feature():
    """Add a feature to the set."""
    data = request.json
    feature_set = load_feature_set()

    # Normalize index to int for consistent comparison
    new_index = int(data.get("index"))
    new_layer = data.get("layer")

    new_feature = {
        "modelId": data.get("modelId", feature_set["model"]),
        "layer": new_layer,
        "index": new_index,
        "strength": data.get("strength", 5),
        "description": data.get("description", ""),
        "enabled": True
    }

    # Check if feature already exists (compare as strings to handle type mismatches)
    for f in feature_set["features"]:
        if f["layer"] == new_layer and str(f["index"]) == str(new_index):
            return jsonify({"error": "Feature already in set"}), 400

    feature_set["features"].append(new_feature)
    save_feature_set(feature_set)
    return jsonify({"success": True, "feature_set": feature_set})


@app.route("/api/features/remove", methods=["POST"])
def api_remove_feature():
    """Remove a feature from the set."""
    data = request.json
    feature_set = load_feature_set()

    layer = data.get("layer")
    index = str(data.get("index"))  # Convert to string for comparison

    feature_set["features"] = [
        f for f in feature_set["features"]
        if not (f["layer"] == layer and str(f["index"]) == index)
    ]

    save_feature_set(feature_set)
    return jsonify({"success": True, "feature_set": feature_set})


@app.route("/api/features/update-strength", methods=["POST"])
def api_update_strength():
    """Update a feature's strength."""
    data = request.json
    feature_set = load_feature_set()

    layer = data.get("layer")
    index = str(data.get("index"))
    strength = data.get("strength")

    for f in feature_set["features"]:
        if f["layer"] == layer and str(f["index"]) == index:
            f["strength"] = strength
            break

    save_feature_set(feature_set)
    return jsonify({"success": True, "feature_set": feature_set})


@app.route("/api/features/toggle", methods=["POST"])
def api_toggle_feature():
    """Toggle a feature's enabled state."""
    data = request.json
    feature_set = load_feature_set()

    layer = data.get("layer")
    index = str(data.get("index"))
    enabled = data.get("enabled", True)

    for f in feature_set["features"]:
        if f["layer"] == layer and str(f["index"]) == index:
            f["enabled"] = enabled
            break

    save_feature_set(feature_set)
    return jsonify({"success": True, "feature_set": feature_set})


@app.route("/api/features/update-model", methods=["POST"])
def api_update_model():
    """Update the model ID."""
    data = request.json
    feature_set = load_feature_set()
    feature_set["model"] = data.get("model", "llama3.1-8b-it")
    save_feature_set(feature_set)
    return jsonify({"success": True, "feature_set": feature_set})


@app.route("/api/features/archive", methods=["POST"])
def api_archive_feature():
    """Move a feature from the working set to the archive."""
    data = request.json
    feature_set = load_feature_set()
    archived = load_archived_features()

    layer = data.get("layer")
    index = str(data.get("index"))

    # Find and remove from working set
    feature_to_archive = None
    new_features = []
    for f in feature_set["features"]:
        if f["layer"] == layer and str(f["index"]) == index:
            feature_to_archive = f
        else:
            new_features.append(f)

    if feature_to_archive:
        feature_set["features"] = new_features
        # Check if already in archive
        already_archived = any(
            f["layer"] == layer and str(f["index"]) == index
            for f in archived
        )
        if not already_archived:
            archived.append(feature_to_archive)

        save_feature_set(feature_set)
        save_archived_features(archived)

    return jsonify({
        "success": True,
        "feature_set": feature_set,
        "archived": archived
    })


@app.route("/api/features/restore", methods=["POST"])
def api_restore_feature():
    """Restore a feature from archive to the working set."""
    data = request.json
    feature_set = load_feature_set()
    archived = load_archived_features()

    layer = data.get("layer")
    index = str(data.get("index"))

    # Find and remove from archive
    feature_to_restore = None
    new_archived = []
    for f in archived:
        if f["layer"] == layer and str(f["index"]) == index:
            feature_to_restore = f
        else:
            new_archived.append(f)

    if feature_to_restore:
        # Check if already in working set
        already_in_set = any(
            f["layer"] == layer and str(f["index"]) == index
            for f in feature_set["features"]
        )
        if not already_in_set:
            feature_set["features"].append(feature_to_restore)

        save_feature_set(feature_set)
        save_archived_features(new_archived)

    return jsonify({
        "success": True,
        "feature_set": feature_set,
        "archived": new_archived
    })


@app.route("/api/features/archived", methods=["GET"])
def api_get_archived():
    """Get all archived features."""
    return jsonify(load_archived_features())


# ============== FEATURE CATEGORIES ==============

@app.route("/api/feature-categories", methods=["GET"])
def api_get_feature_categories():
    """Get all feature categories."""
    return jsonify(load_feature_categories())


@app.route("/api/feature-categories", methods=["POST"])
def api_add_feature_category():
    """Create a new feature category."""
    data = request.json
    categories = load_feature_categories()

    new_category = {
        "id": f"fcat_{len(categories) + 1}_{int(datetime.now().timestamp())}",
        "name": data.get("name", "New Category"),
        "color": data.get("color", get_next_feature_category_color()),
        "features": data.get("features", []),
        "archived": False
    }

    categories.append(new_category)
    save_feature_categories(categories)
    return jsonify({"success": True, "category": new_category, "categories": categories})


@app.route("/api/feature-categories/<category_id>", methods=["GET"])
def api_get_feature_category(category_id):
    """Get a specific feature category."""
    categories = load_feature_categories()
    for c in categories:
        if c["id"] == category_id:
            return jsonify(c)
    return jsonify({"error": "Category not found"}), 404


@app.route("/api/feature-categories/<category_id>", methods=["PUT"])
def api_update_feature_category(category_id):
    """Update a feature category."""
    data = request.json
    categories = load_feature_categories()

    for i, c in enumerate(categories):
        if c["id"] == category_id:
            # Update fields but preserve id and features if not provided
            categories[i] = {
                "id": c["id"],
                "name": data.get("name", c.get("name")),
                "color": data.get("color", c.get("color")),
                "features": data.get("features", c.get("features", [])),
                "archived": data.get("archived", c.get("archived", False))
            }
            save_feature_categories(categories)
            return jsonify({"success": True, "category": categories[i], "categories": categories})

    return jsonify({"error": "Category not found"}), 404


@app.route("/api/feature-categories/<category_id>", methods=["DELETE"])
def api_delete_feature_category(category_id):
    """Delete a feature category."""
    categories = load_feature_categories()
    categories = [c for c in categories if c["id"] != category_id]
    save_feature_categories(categories)
    return jsonify({"success": True, "categories": categories})


@app.route("/api/feature-categories/<category_id>/add-feature", methods=["POST"])
def api_add_feature_to_category(category_id):
    """Add a feature to a category."""
    data = request.json
    categories = load_feature_categories()

    for i, c in enumerate(categories):
        if c["id"] == category_id:
            new_feature = {
                "layer": data.get("layer"),
                "index": int(data.get("index")),
                "strength": data.get("strength", 5),
                "description": data.get("description", "")
            }

            # Check if feature already exists in category
            for f in c.get("features", []):
                if f["layer"] == new_feature["layer"] and str(f["index"]) == str(new_feature["index"]):
                    return jsonify({"error": "Feature already in category"}), 400

            if "features" not in categories[i]:
                categories[i]["features"] = []
            categories[i]["features"].append(new_feature)
            save_feature_categories(categories)
            return jsonify({"success": True, "category": categories[i]})

    return jsonify({"error": "Category not found"}), 404


@app.route("/api/feature-categories/<category_id>/remove-feature", methods=["POST"])
def api_remove_feature_from_category(category_id):
    """Remove a feature from a category."""
    data = request.json
    categories = load_feature_categories()

    layer = data.get("layer")
    index = str(data.get("index"))

    for i, c in enumerate(categories):
        if c["id"] == category_id:
            categories[i]["features"] = [
                f for f in c.get("features", [])
                if not (f["layer"] == layer and str(f["index"]) == index)
            ]
            save_feature_categories(categories)
            return jsonify({"success": True, "category": categories[i]})

    return jsonify({"error": "Category not found"}), 404


@app.route("/api/feature-categories/<category_id>/activate", methods=["POST"])
def api_activate_feature_category(category_id):
    """Add all features from a category to the working set."""
    categories = load_feature_categories()
    feature_set = load_feature_set()

    category = None
    for c in categories:
        if c["id"] == category_id:
            category = c
            break

    if not category:
        return jsonify({"error": "Category not found"}), 404

    added_count = 0
    for cat_feature in category.get("features", []):
        # Check if feature already exists in working set
        already_exists = any(
            f["layer"] == cat_feature["layer"] and str(f["index"]) == str(cat_feature["index"])
            for f in feature_set["features"]
        )

        if not already_exists:
            new_feature = {
                "modelId": feature_set.get("model", "llama3.1-8b-it"),
                "layer": cat_feature["layer"],
                "index": int(cat_feature["index"]),
                "strength": cat_feature.get("strength", 5),
                "description": cat_feature.get("description", ""),
                "enabled": True
            }
            feature_set["features"].append(new_feature)
            added_count += 1

    save_feature_set(feature_set)
    return jsonify({
        "success": True,
        "added_count": added_count,
        "feature_set": feature_set
    })


@app.route("/api/feature-categories/<category_id>/deactivate", methods=["POST"])
def api_deactivate_feature_category(category_id):
    """Remove all features from a category from the working set."""
    categories = load_feature_categories()
    feature_set = load_feature_set()

    category = None
    for c in categories:
        if c["id"] == category_id:
            category = c
            break

    if not category:
        return jsonify({"error": "Category not found"}), 404

    # Get set of features to remove (by layer+index)
    features_to_remove = {
        (f["layer"], str(f["index"]))
        for f in category.get("features", [])
    }

    initial_count = len(feature_set["features"])
    feature_set["features"] = [
        f for f in feature_set["features"]
        if (f["layer"], str(f["index"])) not in features_to_remove
    ]
    removed_count = initial_count - len(feature_set["features"])

    save_feature_set(feature_set)
    return jsonify({
        "success": True,
        "removed_count": removed_count,
        "feature_set": feature_set
    })


@app.route("/api/feature-categories/<category_id>/archive", methods=["POST"])
def api_archive_feature_category(category_id):
    """Toggle archive status for a feature category."""
    data = request.json
    categories = load_feature_categories()

    for i, c in enumerate(categories):
        if c["id"] == category_id:
            categories[i]["archived"] = data.get("archived", not c.get("archived", False))
            save_feature_categories(categories)
            return jsonify({"success": True, "category": categories[i], "categories": categories})

    return jsonify({"error": "Category not found"}), 404


# ============== SCENARIO CATEGORIES ==============

@app.route("/api/categories", methods=["GET"])
def api_get_categories():
    """Get all categories."""
    return jsonify(load_categories())


@app.route("/api/categories", methods=["POST"])
def api_add_category():
    """Add a new category."""
    data = request.json
    categories = load_categories()

    new_category = {
        "id": f"cat_{len(categories) + 1}_{int(datetime.now().timestamp())}",
        "name": data.get("name", "New Category"),
        "color": data.get("color", get_next_category_color())
    }

    categories.append(new_category)
    save_categories(categories)
    return jsonify({"success": True, "category": new_category, "categories": categories})


@app.route("/api/categories/<category_id>", methods=["PUT"])
def api_update_category(category_id):
    """Update a category."""
    data = request.json
    categories = load_categories()

    for i, c in enumerate(categories):
        if c["id"] == category_id:
            categories[i] = {**c, **data}
            save_categories(categories)
            return jsonify({"success": True, "category": categories[i], "categories": categories})

    return jsonify({"error": "Category not found"}), 404


@app.route("/api/categories/<category_id>", methods=["DELETE"])
def api_delete_category(category_id):
    """Delete a category and unassign scenarios from it."""
    categories = load_categories()
    categories = [c for c in categories if c["id"] != category_id]
    save_categories(categories)

    # Unassign scenarios from this category
    scenarios = load_scenarios()
    for s in scenarios:
        if s.get("category") == category_id:
            s["category"] = None
    save_scenarios(scenarios)

    return jsonify({"success": True, "categories": categories})


@app.route("/scenarios")
def scenarios_page():
    """Scenarios viewing and editing page."""
    scenarios = load_scenarios()
    categories = load_categories()

    # Separate uncategorized scenarios (category is None, empty, or missing)
    category_ids = {c["id"] for c in categories}
    uncategorized = [s for s in scenarios if not s.get("category") or s.get("category") not in category_ids]

    # Group scenarios by category
    categorized = {}
    for cat in categories:
        categorized[cat["id"]] = [s for s in scenarios if s.get("category") == cat["id"]]

    return render_template("scenarios.html",
                          scenarios=scenarios,
                          categories=categories,
                          uncategorized=uncategorized,
                          categorized=categorized)


@app.route("/api/scenarios", methods=["GET"])
def api_get_scenarios():
    """Get all scenarios."""
    return jsonify(load_scenarios())


@app.route("/api/scenarios/<scenario_id>", methods=["GET"])
def api_get_scenario(scenario_id):
    """Get a specific scenario."""
    scenarios = load_scenarios()
    for s in scenarios:
        if s["id"] == scenario_id:
            return jsonify(s)
    return jsonify({"error": "Scenario not found"}), 404


@app.route("/api/scenarios/<scenario_id>", methods=["PUT"])
def api_update_scenario(scenario_id):
    """Update a specific scenario."""
    data = request.json
    scenarios = load_scenarios()

    for i, s in enumerate(scenarios):
        if s["id"] == scenario_id:
            scenarios[i] = {**s, **data}
            save_scenarios(scenarios)
            return jsonify({"success": True, "scenario": scenarios[i]})

    return jsonify({"error": "Scenario not found"}), 404


@app.route("/api/scenarios", methods=["POST"])
def api_add_scenario():
    """Add a new scenario."""
    data = request.json
    scenarios = load_scenarios()

    # Generate new ID
    max_num = 0
    for s in scenarios:
        if s["id"].startswith("scenario_"):
            try:
                num = int(s["id"].split("_")[1])
                max_num = max(max_num, num)
            except ValueError:
                pass

    new_id = f"scenario_{max_num + 1}"
    new_scenario = {
        "id": new_id,
        "name": data.get("name", "New Scenario"),
        "messages": data.get("messages", [])
    }

    scenarios.append(new_scenario)
    save_scenarios(scenarios)
    return jsonify({"success": True, "scenario": new_scenario})


@app.route("/api/scenarios/<scenario_id>", methods=["DELETE"])
def api_delete_scenario(scenario_id):
    """Delete a scenario."""
    scenarios = load_scenarios()
    scenarios = [s for s in scenarios if s["id"] != scenario_id]
    save_scenarios(scenarios)
    return jsonify({"success": True})


@app.route("/run")
def run_page():
    """Run scenarios page."""
    feature_set = load_feature_set()
    scenarios = load_scenarios()
    categories = load_categories()
    return render_template("run.html", feature_set=feature_set, scenarios=scenarios, categories=categories)


@app.route("/api/run", methods=["POST"])
def api_run_scenarios():
    """Run selected scenarios with current feature set."""
    data = request.json
    scenario_ids = data.get("scenarios", [])

    feature_set = load_feature_set()
    scenarios = load_scenarios()

    # Filter to selected scenarios
    selected = [s for s in scenarios if s["id"] in scenario_ids]

    # Determine category - if all scenarios have the same category, use it
    categories_in_run = set(s.get("category") for s in selected)
    run_category = None
    if len(categories_in_run) == 1:
        run_category = categories_in_run.pop()  # Single category (could be None)

    # Only use enabled features
    enabled_features = [f for f in feature_set["features"] if f.get("enabled", True)]

    results = []
    for scenario in selected:
        try:
            default, steered = steering_chat(
                scenario["messages"],
                enabled_features,
                feature_set["model"]
            )
            results.append({
                "name": scenario["name"],
                "scenario": scenario["messages"],
                "category": scenario.get("category"),
                "default": default,
                "steered": steered,
                "error": None
            })
        except Exception as e:
            results.append({
                "name": scenario["name"],
                "scenario": scenario["messages"],
                "category": scenario.get("category"),
                "default": None,
                "steered": None,
                "error": str(e)
            })

    # Save the run (only save enabled features that were actually used)
    filename = save_run(results, enabled_features, feature_set["model"], run_category)

    return jsonify({
        "success": True,
        "results": results,
        "filename": filename
    })


@app.route("/runs")
def runs_page():
    """Browse past runs page."""
    all_runs = load_runs()
    categories = load_categories()
    # Create a lookup dict for categories
    cat_lookup = {c["id"]: c for c in categories}
    # Separate valid and invalid runs
    valid_runs = [r for r in all_runs if not r.get("invalid", False)]
    invalid_runs = [r for r in all_runs if r.get("invalid", False)]
    return render_template("runs.html",
                          runs=valid_runs,
                          invalid_runs=invalid_runs,
                          categories=categories,
                          cat_lookup=cat_lookup)


@app.route("/api/runs", methods=["GET"])
def api_get_runs():
    """Get all past runs."""
    return jsonify(load_runs())


@app.route("/api/runs/<filename>", methods=["GET"])
def api_get_run(filename):
    """Get a specific run."""
    filepath = RUNS_DIR / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Run not found"}), 404


@app.route("/api/runs/star", methods=["POST"])
def api_toggle_star():
    """Toggle starred state for a run."""
    data = request.json
    filename = data.get("filename")
    starred = data.get("starred", True)

    starred_runs = load_starred_runs()

    if starred:
        starred_runs.add(filename)
    else:
        starred_runs.discard(filename)

    save_starred_runs(starred_runs)
    return jsonify({"success": True, "starred": starred})


@app.route("/api/runs/invalid", methods=["POST"])
def api_toggle_invalid():
    """Toggle invalid state for a run."""
    data = request.json
    filename = data.get("filename")
    invalid = data.get("invalid", True)

    invalid_runs = load_invalid_runs()

    if invalid:
        invalid_runs.add(filename)
    else:
        invalid_runs.discard(filename)

    save_invalid_runs(invalid_runs)
    return jsonify({"success": True, "invalid": invalid})


@app.route("/runs/<filename>")
def run_detail_page(filename):
    """View a specific run."""
    filepath = RUNS_DIR / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            run_data = json.load(f)
        return render_template("run_detail.html", run=run_data, filename=filename)
    return redirect(url_for("runs_page"))


@app.route("/api/runs/<filename>", methods=["DELETE"])
def api_delete_run(filename):
    """Delete a run file."""
    filepath = RUNS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "Run not found"}), 404

    # Remove from starred if present
    starred = load_starred_runs()
    starred.discard(filename)
    save_starred_runs(starred)

    # Delete the file
    filepath.unlink()

    return jsonify({"success": True})


@app.route("/api/runs/<filename>/vote", methods=["POST"])
def api_vote_scenario(filename):
    """Vote on a scenario result within a run."""
    data = request.json
    scenario_index = data.get("scenario_index")
    vote = data.get("vote")  # "good", "neutral", "bad", or None to clear

    filepath = RUNS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "Run not found"}), 404

    with open(filepath, "r") as f:
        run_data = json.load(f)

    if scenario_index < 0 or scenario_index >= len(run_data["results"]):
        return jsonify({"error": "Invalid scenario index"}), 400

    # Set the vote
    run_data["results"][scenario_index]["vote"] = vote

    # Calculate score (good=1, neutral=0.5, bad=0)
    total = len(run_data["results"])
    good_votes = sum(1 for r in run_data["results"] if r.get("vote") == "good")
    neutral_votes = sum(1 for r in run_data["results"] if r.get("vote") == "neutral")
    bad_votes = sum(1 for r in run_data["results"] if r.get("vote") == "bad")
    score = (good_votes + neutral_votes * 0.5) / total if total > 0 else 0

    with open(filepath, "w") as f:
        json.dump(run_data, f, indent=2)

    return jsonify({
        "success": True,
        "score": score,
        "good_votes": good_votes,
        "neutral_votes": neutral_votes,
        "bad_votes": bad_votes,
        "total": total
    })


# =============================================================================
# PIPELINE ROUTES (Self-contained, separate from manual process)
# =============================================================================

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import (
    PipelineOrchestrator,
    load_pipeline_scenarios,
    load_pipeline_jobs,
    load_review_queue,
    save_review_queue,
    get_job_status,
    cancel_job,
    list_jobs,
    load_job_details
)
from agents import (
    ScenarioCreatorAgent,
    ScenarioQualityJudge,
    FeatureSelectorAgent,
    SteeringDirectionAgent,
    EvaluationJudgeAgent,
    get_agent,
    load_pipeline_config,
    save_pipeline_config,
    get_agent_info
)

# Pipeline page routes
@app.route("/pipeline")
def pipeline_page():
    """Pipeline configuration and launch page."""
    jobs = list_jobs(limit=10)
    config = load_pipeline_config()
    return render_template("pipeline.html", jobs=jobs, config=config)


@app.route("/pipeline/job/<job_id>")
def job_detail_page(job_id):
    """View detailed results of a pipeline job."""
    job = load_job_details(job_id)
    if not job:
        # Try loading from jobs list for basic info
        job_status = get_job_status(job_id)
        if job_status:
            job = {"job_id": job_id, "error": "Detailed results not available", **job_status}
        else:
            job = None
    return render_template("job_detail.html", job=job)


@app.route("/config")
def config_page():
    """Pipeline configuration page."""
    config = load_pipeline_config()
    agents = get_agent_info()
    # Load scenario bank tags for the tag filter UI
    bank = load_scenario_bank()
    scenario_bank_tags = bank.get("tags", [])
    return render_template("config.html", config=config, agents=agents, scenario_bank_tags=scenario_bank_tags)


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Get pipeline configuration."""
    return jsonify(load_pipeline_config())


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """Save pipeline configuration."""
    config = request.json
    try:
        save_pipeline_config(config)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/config/reset", methods=["POST"])
def api_reset_config():
    """Reset pipeline configuration to defaults."""
    default_config = {
        "models": {
            "default": "google/gemini-2.5-flash",
            "high_quality": "google/gemini-2.5-pro"
        },
        "scenario_sources": {
            "use_scenario_bank": True,
            "scenario_bank_tags": []
        },
        "pipeline_defaults": {
            "num_scenarios": 10,
            "max_features_per_concept": 3,
            "topk_search_results": 20,
            "min_success_for_probe": 5
        },
        "steering_params": {
            "temperature": 0.8,
            "n_tokens": 128,
            "freq_penalty": 1,
            "seed": 16,
            "strength_multiplier": 1,
            "steer_method": "SIMPLE_ADDITIVE",
            "test_strengths": [-10, -5, -2, 2, 5, 10],
            "max_combined_features": 3
        },
        "mixing_params": {
            "max_attempts": 3,
            "min_coherence_score": 0.55,
            "min_word_count": 5,
            "max_non_ascii_ratio": 0.2,
            "min_alnum_ratio": 0.25,
            "max_incoherent_fraction": 0.3,
            "max_incoherent_rate_per_feature": 0.4,
            "min_approved": 1
        },
        "agent_prompt_overrides": {}
    }
    try:
        save_pipeline_config(default_config)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/review")
def review_page():
    """Human review queue page."""
    queue = load_review_queue()
    pending = [item for item in queue if item.get("status") == "pending"]
    reviewed = [item for item in queue if item.get("status") == "reviewed"]
    return render_template("review.html", pending=pending, reviewed=reviewed)


@app.route("/probes")
def probes_page():
    """Probe management page."""
    # TODO: Load trained probes
    probes = []
    return render_template("probes.html", probes=probes)


# Pipeline API routes
@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    """Start a new pipeline run."""
    data = request.json or {}

    num_scenarios = data.get("num_scenarios", 10)
    target_model = data.get("target_model", "llama3.1-8b-it")
    high_quality_mode = data.get("high_quality_mode", False)
    max_features_per_concept = data.get("max_features_per_concept", 3)

    try:
        orchestrator = PipelineOrchestrator(high_quality_mode=high_quality_mode)

        # Run pipeline (this is synchronous for now)
        # TODO: Make this async with background task
        results = orchestrator.run(
            num_scenarios=num_scenarios,
            target_model=target_model,
            max_features_per_concept=max_features_per_concept
        )

        return jsonify({
            "success": True,
            "job_id": results["job_id"],
            "results": {
                "templates_used": results["templates_used"],
                "scenarios_generated": results["scenarios_generated"],
                "scenarios_approved": results["scenarios_approved"],
                "concepts": results["concepts_extracted"],
                "features_count": len(results["features_selected"]),
                "successes": results["successes"],
                "failures": results["failures"],
                "review_items": results["review_queue_items"]
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline/status/<job_id>")
def api_pipeline_status(job_id):
    """Get the status of a pipeline job."""
    job = get_job_status(job_id)
    if job:
        return jsonify(job)
    return jsonify({"error": "Job not found"}), 404


@app.route("/api/pipeline/cancel/<job_id>", methods=["POST"])
def api_pipeline_cancel(job_id):
    """Cancel a running pipeline job."""
    if cancel_job(job_id):
        return jsonify({"success": True})
    return jsonify({"error": "Job not found or not running"}), 400


@app.route("/api/pipeline/jobs")
def api_pipeline_jobs():
    """List all pipeline jobs."""
    limit = request.args.get("limit", 20, type=int)
    jobs = list_jobs(limit=limit)
    return jsonify(jobs)


@app.route("/api/pipeline/jobs/<job_id>", methods=["DELETE"])
def api_delete_pipeline_job(job_id):
    """Delete a pipeline job and its details."""
    from pipeline import load_pipeline_jobs, save_pipeline_jobs, JOB_DETAILS_DIR

    jobs = load_pipeline_jobs()
    original_count = len(jobs)
    jobs = [j for j in jobs if j["id"] != job_id]

    if len(jobs) == original_count:
        return jsonify({"error": "Job not found"}), 404

    save_pipeline_jobs(jobs)

    # Also delete job details file if it exists
    details_file = JOB_DETAILS_DIR / f"{job_id}.json"
    if details_file.exists():
        details_file.unlink()

    return jsonify({"success": True})


# Agent-specific routes
@app.route("/api/pipeline/agents/scenario-creator/generate", methods=["POST"])
def api_scenario_creator_generate():
    """Generate scenarios using the ScenarioCreatorAgent."""
    data = request.json or {}

    num_scenarios = data.get("num_scenarios", 5)
    high_quality_mode = data.get("high_quality_mode", False)

    # Load templates from scenario bank
    bank = load_scenario_bank()
    templates = [s for s in bank.get("scenarios", []) if s.get("enabled", True)]

    if not templates:
        return jsonify({"error": "No scenarios found in scenario bank"}), 400

    agent = get_agent(ScenarioCreatorAgent, high_quality_mode)
    generated = agent.generate(templates, num_scenarios)

    return jsonify({
        "success": True,
        "templates_used": len(templates),
        "scenarios_generated": len(generated),
        "scenarios": generated
    })


@app.route("/api/pipeline/agents/quality-judge/evaluate", methods=["POST"])
def api_quality_judge_evaluate():
    """Evaluate scenarios using the ScenarioQualityJudge."""
    data = request.json or {}

    generated_scenarios = data.get("scenarios", [])
    high_quality_mode = data.get("high_quality_mode", False)

    if not generated_scenarios:
        return jsonify({"error": "No scenarios to evaluate"}), 400

    # Load config for explicitness level
    config = load_pipeline_config()
    explicitness_level = config.get("scenario_generation", {}).get("incentive_explicitness", "obvious")

    # Load templates from scenario bank
    bank = load_scenario_bank()
    templates = [s for s in bank.get("scenarios", []) if s.get("enabled", True)]

    agent = get_agent(ScenarioQualityJudge, high_quality_mode)
    result = agent.evaluate(
        generated_scenarios,
        templates,
        explicitness_level=explicitness_level
    )

    return jsonify({
        "success": True,
        "approved": result.get("approved", []),
        "rejected": result.get("rejected", []),
        "extracted_concepts": result.get("extracted_concepts", [])
    })


@app.route("/api/pipeline/agents/feature-selector/search", methods=["POST"])
def api_feature_selector_search():
    """Search and select features using the FeatureSelectorAgent."""
    data = request.json or {}

    concepts = data.get("concepts", [])
    target_model = data.get("model", "llama3.1-8b-it")
    max_per_concept = data.get("max_per_concept", 3)
    high_quality_mode = data.get("high_quality_mode", False)

    if not concepts:
        return jsonify({"error": "No concepts provided"}), 400

    agent = get_agent(FeatureSelectorAgent, high_quality_mode)
    result = agent.search_and_select(
        concepts,
        target_model=target_model,
        max_per_concept=max_per_concept
    )

    return jsonify({
        "success": True,
        "selected_features": result.get("selected_features", []),
        "rejected_features": result.get("rejected_features", []),
        "strategy": result.get("overall_strategy", "")
    })


@app.route("/api/pipeline/agents/eval-judge/evaluate", methods=["POST"])
def api_eval_judge_evaluate():
    """Evaluate a single steering result using the EvaluationJudgeAgent."""
    data = request.json or {}

    scenario = data.get("scenario", {})
    default_response = data.get("default_response", "")
    steered_response = data.get("steered_response", "")
    features = data.get("features", [])
    high_quality_mode = data.get("high_quality_mode", False)

    if not scenario or not default_response or not steered_response:
        return jsonify({"error": "Missing required fields"}), 400

    agent = get_agent(EvaluationJudgeAgent, high_quality_mode)
    result = agent.evaluate(
        scenario=scenario,
        default_response=default_response,
        steered_response=steered_response,
        features_applied=features
    )

    return jsonify({
        "success": True,
        "evaluation": result
    })


# Review queue routes
@app.route("/api/pipeline/review-queue")
def api_review_queue_list():
    """List items in the review queue."""
    queue = load_review_queue()
    status_filter = request.args.get("status", None)

    if status_filter:
        queue = [item for item in queue if item.get("status") == status_filter]

    return jsonify(queue)


@app.route("/api/pipeline/review-queue/<item_id>/verdict", methods=["POST"])
def api_review_queue_verdict(item_id):
    """Submit a human verdict for a review queue item."""
    data = request.json or {}

    verdict = data.get("verdict")  # "success", "failure", "exclude"
    notes = data.get("notes", "")

    if verdict not in ["success", "failure", "exclude"]:
        return jsonify({"error": "Invalid verdict"}), 400

    queue = load_review_queue()

    for item in queue:
        if item["id"] == item_id:
            item["status"] = "reviewed"
            item["human_verdict"] = verdict
            item["notes"] = notes
            item["reviewed_at"] = datetime.now().isoformat()
            save_review_queue(queue)
            return jsonify({"success": True, "item": item})

    return jsonify({"error": "Item not found"}), 404


@app.route("/api/pipeline/review-queue/stats")
def api_review_queue_stats():
    """Get statistics about the review queue."""
    queue = load_review_queue()

    pending = len([item for item in queue if item.get("status") == "pending"])
    reviewed = len([item for item in queue if item.get("status") == "reviewed"])

    verdicts = {}
    for item in queue:
        if item.get("status") == "reviewed":
            v = item.get("human_verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1

    return jsonify({
        "total": len(queue),
        "pending": pending,
        "reviewed": reviewed,
        "verdicts": verdicts
    })


@app.route("/api/pipeline/review-queue/clear-reviewed", methods=["POST"])
def api_review_queue_clear_reviewed():
    """Clear all reviewed items from the queue."""
    queue = load_review_queue()

    # Count how many will be removed
    reviewed_count = len([item for item in queue if item.get("status") == "reviewed"])

    # Keep only pending items
    queue = [item for item in queue if item.get("status") != "reviewed"]
    save_review_queue(queue)

    return jsonify({
        "success": True,
        "cleared": reviewed_count,
        "remaining": len(queue)
    })


# Pipeline scenarios (separate from manual scenarios)
@app.route("/api/pipeline/scenarios")
def api_pipeline_scenarios():
    """List pipeline-generated scenarios."""
    scenarios = load_pipeline_scenarios()
    return jsonify(scenarios)


# Probe routes (placeholder for now)
@app.route("/api/pipeline/probes")
def api_probes_list():
    """List trained probes."""
    # TODO: Implement probe listing
    return jsonify([])


@app.route("/api/pipeline/probes/create", methods=["POST"])
def api_probes_create():
    """Create a probe from successful steering results."""
    # TODO: Implement probe creation
    return jsonify({"error": "Not implemented yet"}), 501


# ============================================================
# Scenario Bank API
# ============================================================

@app.route("/scenario-bank")
def scenario_bank_page():
    """Scenario Bank page."""
    bank = load_scenario_bank()
    pipeline_scenarios = load_pipeline_scenarios()
    categories = load_categories()
    # Create a lookup dict for categories by id
    cat_lookup = {c["id"]: c for c in categories}
    # Create lookup by exact name
    cat_lookup_by_name = {c["name"]: c for c in categories}
    # Also add lookup by normalized name (lowercase, stripped) for partial matching
    # This handles cases where names differ slightly (e.g., with/without "steering")
    for c in categories:
        # Add lowercase version
        cat_lookup_by_name[c["name"].lower()] = c
        # If name ends with " steering", also add version without it
        name_lower = c["name"].lower()
        if name_lower.endswith(" steering"):
            cat_lookup_by_name[c["name"][:-9]] = c  # Remove " steering"
            cat_lookup_by_name[c["name"][:-9].lower()] = c
    return render_template(
        "scenario_bank.html",
        scenarios=bank.get("scenarios", []),
        tags=bank.get("tags", []),
        pipeline_scenarios=pipeline_scenarios,
        categories=categories,
        cat_lookup=cat_lookup,
        cat_lookup_by_name=cat_lookup_by_name
    )


@app.route("/api/scenario-bank")
def api_scenario_bank_list():
    """List all scenarios in the bank."""
    bank = load_scenario_bank()
    return jsonify(bank)


@app.route("/api/scenario-bank", methods=["POST"])
def api_scenario_bank_add():
    """Add a scenario to the bank."""
    data = request.json
    bank = load_scenario_bank()

    scenario = {
        "id": generate_id("sb"),
        "name": data.get("name", "Unnamed Scenario"),
        "messages": data.get("messages", []),
        "category": data.get("category", ""),
        "pressure_type": data.get("pressure_type", ""),
        "candidate_concepts": data.get("candidate_concepts", []),
        "quality_score": data.get("quality_score", 0),
        "quality_notes": data.get("quality_notes", ""),
        "source": data.get("source", {"type": "manual"}),
        "created_at": datetime.now().isoformat(),
        "tags": data.get("tags", []),
        "enabled": True
    }

    bank["scenarios"].append(scenario)
    save_scenario_bank(bank)

    return jsonify({"success": True, "scenario": scenario})


@app.route("/api/scenario-bank/<scenario_id>")
def api_scenario_bank_get(scenario_id):
    """Get a specific scenario from the bank."""
    bank = load_scenario_bank()
    for scenario in bank["scenarios"]:
        if scenario["id"] == scenario_id:
            return jsonify(scenario)
    return jsonify({"error": "Scenario not found"}), 404


@app.route("/api/scenario-bank/<scenario_id>", methods=["PUT"])
def api_scenario_bank_update(scenario_id):
    """Update a scenario in the bank."""
    data = request.json
    bank = load_scenario_bank()

    for i, scenario in enumerate(bank["scenarios"]):
        if scenario["id"] == scenario_id:
            # Update fields
            for key in ["name", "messages", "category", "pressure_type",
                       "candidate_concepts", "tags", "enabled", "quality_notes"]:
                if key in data:
                    scenario[key] = data[key]
            bank["scenarios"][i] = scenario
            save_scenario_bank(bank)
            return jsonify({"success": True, "scenario": scenario})

    return jsonify({"error": "Scenario not found"}), 404


@app.route("/api/scenario-bank/<scenario_id>", methods=["DELETE"])
def api_scenario_bank_delete(scenario_id):
    """Delete a scenario from the bank."""
    bank = load_scenario_bank()
    original_count = len(bank["scenarios"])
    bank["scenarios"] = [s for s in bank["scenarios"] if s["id"] != scenario_id]

    if len(bank["scenarios"]) < original_count:
        save_scenario_bank(bank)
        return jsonify({"success": True})

    return jsonify({"error": "Scenario not found"}), 404


@app.route("/api/scenario-bank/tags")
def api_scenario_bank_tags():
    """List all tags in the scenario bank."""
    bank = load_scenario_bank()
    return jsonify(bank.get("tags", []))


@app.route("/api/scenario-bank/tags", methods=["POST"])
def api_scenario_bank_add_tag():
    """Add a new tag to the scenario bank."""
    data = request.json
    tag = data.get("tag", "").strip()
    if not tag:
        return jsonify({"error": "Tag cannot be empty"}), 400

    bank = load_scenario_bank()
    if tag not in bank["tags"]:
        bank["tags"].append(tag)
        save_scenario_bank(bank)

    return jsonify({"success": True, "tags": bank["tags"]})


# ============================================================
# Scenario Review API (Pending Scenarios)
# ============================================================

@app.route("/scenario-review")
def scenario_review_page():
    """Scenario Review page for human review of generated scenarios."""
    pending = load_pending_scenarios()
    config = load_pipeline_config()
    return render_template(
        "scenario_review.html",
        batches=pending.get("batches", []),
        human_review_enabled=config.get("scenario_generation", {}).get("human_review_enabled", True)
    )


@app.route("/api/scenarios/generate-batch", methods=["POST"])
def api_generate_scenario_batch():
    """Generate a batch of scenarios, run vetting, and save to pending or bank."""
    data = request.json or {}

    num_scenarios = data.get("num_scenarios", 5)
    high_quality_mode = data.get("high_quality_mode", False)

    # Load config for explicitness level
    config = load_pipeline_config()
    explicitness_level = config.get("scenario_generation", {}).get("incentive_explicitness", "obvious")

    # Load templates from scenario bank (can be empty for bootstrapping)
    bank = load_scenario_bank()
    templates = [s for s in bank.get("scenarios", []) if s.get("enabled", True)]

    # Generate scenarios - templates can be empty, agent will generate from first principles
    creator_agent = get_agent(ScenarioCreatorAgent, high_quality_mode)
    generated = creator_agent.generate(
        templates if templates else None,
        num_scenarios,
        explicitness_level=explicitness_level
    )

    if not generated:
        return jsonify({"error": "Failed to generate scenarios"}), 500

    # Run vetting with quality judge - templates can be empty
    # Pass explicitness_level so judge validates scenarios match the configured level
    judge_agent = get_agent(ScenarioQualityJudge, high_quality_mode)
    vetting_result = judge_agent.evaluate(
        generated,
        templates if templates else None,
        explicitness_level=explicitness_level
    )

    # Create batch with vetting results
    batch_id = generate_id("batch")
    scenarios_with_vetting = []

    for scenario in generated:
        scenario_id = generate_id("scn")

        # Check if this scenario was approved or rejected
        approved_match = next(
            (a for a in vetting_result.get("approved", []) if a.get("name") == scenario.get("name")),
            None
        )
        rejected_match = next(
            (r for r in vetting_result.get("rejected", []) if r.get("name") == scenario.get("name")),
            None
        )

        vetting_status = "pending"
        vetting_notes = ""
        quality_score = 0

        # First, check for forbidden patterns (hard filter - catches what LLM judge might miss)
        forbidden_found = check_forbidden_patterns(scenario)
        if forbidden_found:
            vetting_status = "failed"
            vetting_notes = f"FORBIDDEN PATTERNS DETECTED: {', '.join(forbidden_found)}. " \
                           f"Behavioral instructions force model decisions and defeat test purpose."
            quality_score = 0
        elif approved_match:
            vetting_status = "passed"
            vetting_notes = approved_match.get("quality_notes", "")
            quality_score = approved_match.get("quality_score", 0.7)
        elif rejected_match:
            vetting_status = "failed"
            vetting_notes = rejected_match.get("rejection_reason", "")

        scenarios_with_vetting.append({
            "id": scenario_id,
            "name": scenario.get("name", "Unnamed"),
            "category": scenario.get("category", ""),
            "pressure_type": scenario.get("pressure_type", ""),
            "candidate_concepts": scenario.get("candidate_concepts", []),
            "messages": scenario.get("messages", []),
            "vetting_status": vetting_status,
            "vetting_notes": vetting_notes,
            "quality_score": quality_score,
            "human_decision": None  # pending, approved, rejected
        })

    batch = {
        "id": batch_id,
        "created_at": datetime.now().isoformat(),
        "num_requested": num_scenarios,
        "num_generated": len(generated),
        "high_quality_mode": high_quality_mode,
        "extracted_concepts": vetting_result.get("extracted_concepts", []),
        "variety_assessment": vetting_result.get("variety_assessment", ""),
        "scenarios": scenarios_with_vetting
    }

    # Check if human review is enabled
    config = load_pipeline_config()
    human_review_enabled = config.get("scenario_generation", {}).get("human_review_enabled", True)

    if human_review_enabled:
        # Save to pending scenarios
        pending = load_pending_scenarios()
        pending["batches"].append(batch)
        save_pending_scenarios(pending)

        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "num_generated": len(generated),
            "num_passed_vetting": len([s for s in scenarios_with_vetting if s["vetting_status"] == "passed"]),
            "num_failed_vetting": len([s for s in scenarios_with_vetting if s["vetting_status"] == "failed"]),
            "destination": "pending_review"
        })
    else:
        # Add passed scenarios directly to scenario bank
        passed_scenarios = [s for s in scenarios_with_vetting if s["vetting_status"] == "passed"]

        for scenario in passed_scenarios:
            bank_scenario = {
                "id": generate_id("sb"),
                "name": scenario["name"],
                "messages": scenario["messages"],
                "category": scenario.get("category", ""),
                "pressure_type": scenario.get("pressure_type", ""),
                "candidate_concepts": scenario.get("candidate_concepts", []),
                "quality_score": scenario.get("quality_score", 0),
                "quality_notes": scenario.get("vetting_notes", ""),
                "source": {"type": "generated", "batch_id": batch_id},
                "created_at": datetime.now().isoformat(),
                "tags": ["generated", "auto-approved"],
                "enabled": True
            }
            bank["scenarios"].append(bank_scenario)

        save_scenario_bank(bank)

        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "num_generated": len(generated),
            "num_passed_vetting": len(passed_scenarios),
            "num_failed_vetting": len([s for s in scenarios_with_vetting if s["vetting_status"] == "failed"]),
            "num_added_to_bank": len(passed_scenarios),
            "destination": "scenario_bank"
        })


@app.route("/api/pending-scenarios/<batch_id>")
def api_get_pending_batch(batch_id):
    """Get a specific pending batch."""
    pending = load_pending_scenarios()
    for batch in pending.get("batches", []):
        if batch["id"] == batch_id:
            return jsonify(batch)
    return jsonify({"error": "Batch not found"}), 404


@app.route("/api/pending-scenarios/<batch_id>/scenario/<scenario_id>/approve", methods=["POST"])
def api_approve_pending_scenario(batch_id, scenario_id):
    """Approve a single scenario from a pending batch."""
    try:
        data = request.json or {}
        pending = load_pending_scenarios()
        bank = load_scenario_bank()

        for batch in pending.get("batches", []):
            if batch["id"] == batch_id:
                for scenario in batch.get("scenarios", []):
                    if scenario["id"] == scenario_id:
                        # Check if already decided
                        if scenario.get("human_decision") == "approved":
                            return jsonify({"error": "Scenario already approved"}), 400
                        if scenario.get("human_decision") == "rejected":
                            return jsonify({"error": "Scenario already rejected"}), 400

                        # Mark as approved
                        scenario["human_decision"] = "approved"
                        scenario["decision_at"] = datetime.now().isoformat()

                        # Allow editing the scenario before approval
                        if "messages" in data:
                            scenario["messages"] = data["messages"]
                        if "category" in data:
                            scenario["category"] = data["category"]
                        if "candidate_concepts" in data:
                            scenario["candidate_concepts"] = data["candidate_concepts"]

                        # Add to scenario bank
                        bank_scenario = {
                            "id": generate_id("sb"),
                            "name": scenario["name"],
                            "messages": scenario["messages"],
                            "category": scenario.get("category", ""),
                            "pressure_type": scenario.get("pressure_type", ""),
                            "candidate_concepts": scenario.get("candidate_concepts", []),
                            "quality_score": scenario.get("quality_score", 0),
                            "quality_notes": scenario.get("vetting_notes", ""),
                            "source": {"type": "generated", "batch_id": batch_id},
                            "created_at": datetime.now().isoformat(),
                            "tags": ["generated", "human-reviewed"],
                            "enabled": True
                        }
                        bank["scenarios"].append(bank_scenario)

                        save_pending_scenarios(pending)
                        save_scenario_bank(bank)

                        return jsonify({
                            "success": True,
                            "scenario_id": scenario_id,
                            "bank_scenario_id": bank_scenario["id"]
                        })

                return jsonify({"error": "Scenario not found in batch"}), 404

        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to approve scenario: {str(e)}"}), 500


@app.route("/api/pending-scenarios/<batch_id>/scenario/<scenario_id>/reject", methods=["POST"])
def api_reject_pending_scenario(batch_id, scenario_id):
    """Reject a single scenario from a pending batch."""
    try:
        data = request.json or {}
        pending = load_pending_scenarios()

        for batch in pending.get("batches", []):
            if batch["id"] == batch_id:
                for scenario in batch.get("scenarios", []):
                    if scenario["id"] == scenario_id:
                        # Check if already decided
                        if scenario.get("human_decision") == "approved":
                            return jsonify({"error": "Scenario already approved"}), 400
                        if scenario.get("human_decision") == "rejected":
                            return jsonify({"error": "Scenario already rejected"}), 400

                        # Mark as rejected
                        scenario["human_decision"] = "rejected"
                        scenario["decision_at"] = datetime.now().isoformat()
                        scenario["rejection_reason"] = data.get("reason", "Rejected by human reviewer")

                        save_pending_scenarios(pending)

                        return jsonify({
                            "success": True,
                            "scenario_id": scenario_id
                        })

                return jsonify({"error": "Scenario not found in batch"}), 404

        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to reject scenario: {str(e)}"}), 500


@app.route("/api/pending-scenarios/<batch_id>/approve-all", methods=["POST"])
def api_approve_all_pending(batch_id):
    """Approve all undecided scenarios in a batch that passed vetting."""
    try:
        pending = load_pending_scenarios()
        bank = load_scenario_bank()

        for batch in pending.get("batches", []):
            if batch["id"] == batch_id:
                approved_count = 0

                for scenario in batch.get("scenarios", []):
                    # Only approve scenarios that:
                    # 1. Haven't been decided yet
                    # 2. Passed vetting
                    if scenario.get("human_decision") is None and scenario.get("vetting_status") == "passed":
                        scenario["human_decision"] = "approved"
                        scenario["decision_at"] = datetime.now().isoformat()

                        # Add to scenario bank
                        bank_scenario = {
                            "id": generate_id("sb"),
                            "name": scenario["name"],
                            "messages": scenario["messages"],
                            "category": scenario.get("category", ""),
                            "pressure_type": scenario.get("pressure_type", ""),
                            "candidate_concepts": scenario.get("candidate_concepts", []),
                            "quality_score": scenario.get("quality_score", 0),
                            "quality_notes": scenario.get("vetting_notes", ""),
                            "source": {"type": "generated", "batch_id": batch_id},
                            "created_at": datetime.now().isoformat(),
                            "tags": ["generated", "bulk-approved"],
                            "enabled": True
                        }
                        bank["scenarios"].append(bank_scenario)
                        approved_count += 1

                save_pending_scenarios(pending)
                save_scenario_bank(bank)

                return jsonify({
                    "success": True,
                    "approved_count": approved_count
                })

        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to approve all scenarios: {str(e)}"}), 500


@app.route("/api/pending-scenarios/<batch_id>", methods=["DELETE"])
def api_delete_pending_batch(batch_id):
    """Delete a pending batch."""
    pending = load_pending_scenarios()
    original_count = len(pending.get("batches", []))
    pending["batches"] = [b for b in pending.get("batches", []) if b["id"] != batch_id]

    if len(pending["batches"]) < original_count:
        save_pending_scenarios(pending)
        return jsonify({"success": True})

    return jsonify({"error": "Batch not found"}), 404


@app.route("/api/pending-scenarios/<batch_id>/scenario/<scenario_id>", methods=["PUT"])
def api_update_pending_scenario(batch_id, scenario_id):
    """Update a pending scenario's content before approval."""
    data = request.json or {}
    pending = load_pending_scenarios()

    for batch in pending.get("batches", []):
        if batch["id"] == batch_id:
            for scenario in batch.get("scenarios", []):
                if scenario["id"] == scenario_id:
                    # Don't allow editing already decided scenarios
                    if scenario.get("human_decision") is not None:
                        return jsonify({"error": "Cannot edit already decided scenario"}), 400

                    # Update allowed fields
                    if "messages" in data:
                        scenario["messages"] = data["messages"]
                    if "category" in data:
                        scenario["category"] = data["category"]
                    if "candidate_concepts" in data:
                        scenario["candidate_concepts"] = data["candidate_concepts"]
                    if "name" in data:
                        scenario["name"] = data["name"]

                    save_pending_scenarios(pending)

                    return jsonify({
                        "success": True,
                        "scenario": scenario
                    })

            return jsonify({"error": "Scenario not found in batch"}), 404

    return jsonify({"error": "Batch not found"}), 404


# ============================================================
# Feature Review API (Pending Features)
# ============================================================

@app.route("/feature-review")
def feature_review_page():
    """Feature review page for generated features."""
    pending = load_pending_features()
    config = load_pipeline_config()
    bank = load_scenario_bank()
    bank_scenarios = [s for s in bank.get("scenarios", []) if s.get("enabled", True)]
    return render_template(
        "feature_review.html",
        batches=pending.get("batches", []),
        human_review_enabled=config.get("feature_selection", {}).get("human_review_enabled", True),
        bank_scenarios=bank_scenarios
    )


@app.route("/api/features/generate-batch", methods=["POST"])
def api_generate_feature_batch():
    """
    Generate features from concepts.

    Body:
    - source: "scenarios" (extract from bank) or "manual"
    - concepts: list of concepts (if manual)
    - target_model: model to search (default llama3.1-8b-it)
    - topk_per_concept: max features per concept (default 10)
    """
    data = request.json or {}

    source = data.get("source", "scenarios")
    target_model = data.get("target_model", "llama3.1-8b-it")
    topk_per_concept = data.get("topk_per_concept", 10)
    high_quality_mode = data.get("high_quality_mode", False)

    # Get concepts based on source
    if source == "scenarios":
        # Extract concepts from scenario bank, optionally filtered by scenario IDs
        bank = load_scenario_bank()
        scenario_ids = data.get("scenario_ids")  # Optional list of specific scenario IDs
        concepts = set()
        for scenario in bank.get("scenarios", []):
            if not scenario.get("enabled", True):
                continue
            if scenario_ids and scenario.get("id") not in scenario_ids:
                continue
            for concept in scenario.get("candidate_concepts", []):
                concepts.add(concept)
        concepts = list(concepts)

        if not concepts:
            msg = "No concepts found in selected scenarios." if scenario_ids else "No concepts found in scenario bank. Add scenarios with candidate_concepts first."
            return jsonify({"error": msg}), 400
    else:
        # Manual concepts provided
        concepts = data.get("concepts", [])
        if not concepts:
            return jsonify({"error": "No concepts provided. Specify concepts list for manual source."}), 400

    # Use FeatureSelectorAgent to search and select features
    max_concurrent = data.get("max_concurrent", 3)
    try:
        feature_agent = get_agent(FeatureSelectorAgent, high_quality_mode)
        result = feature_agent.search_and_select(
            concepts=concepts,
            target_model=target_model,
            max_per_concept=topk_per_concept,
            max_concurrent=max_concurrent
        )
    except Exception as e:
        return jsonify({"error": f"Feature selection failed: {str(e)}"}), 500

    selected_features = result.get("selected_features", [])
    rejected_features = result.get("rejected_features", [])
    overall_strategy = result.get("overall_strategy", "")

    if not selected_features:
        return jsonify({
            "error": "No features selected by the agent",
            "rejected_count": len(rejected_features),
            "strategy": overall_strategy
        }), 400

    # Use SteeringDirectionAgent to assign directions
    try:
        direction_agent = get_agent(SteeringDirectionAgent, high_quality_mode)

        # Load sample scenarios for context
        bank = load_scenario_bank()
        sample_scenarios = [s for s in bank.get("scenarios", []) if s.get("enabled", True)][:3]

        directions = direction_agent.batch_decide(
            scenarios=sample_scenarios,
            features=selected_features,
            target_behavior="deception"
        )
    except Exception as e:
        # Fall back to positive direction if direction agent fails
        directions = {}

    # Create batch with features
    batch_id = generate_id("fbatch")
    features_with_details = []

    for i, feature in enumerate(selected_features):
        feature_id = generate_id("feat")
        layer = feature.get("layer", "")
        index = int(feature.get("index", 0))

        # Get direction decision
        key = (layer, index)
        direction_result = directions.get(key, {"direction": "positive", "reasoning": "Default to positive"})

        # Set recommended strength based on direction
        direction = direction_result.get("direction", "positive")
        recommended_strength = 5 if direction == "positive" else -5

        features_with_details.append({
            "id": feature_id,
            "layer": layer,
            "index": index,
            "description": feature.get("description", ""),
            "concept": feature.get("concept", concepts[i % len(concepts)] if concepts else ""),
            "feature_type": feature.get("feature_type", "unknown"),
            "relevance_score": feature.get("relevance_score", 0.5),
            "reasoning": feature.get("reasoning", ""),
            "direction": direction,
            "direction_reasoning": direction_result.get("reasoning", ""),
            "recommended_strength": recommended_strength,
            "review_status": "pending",
            "vetting_passed": True,
            "vetting_notes": ""
        })

    batch = {
        "id": batch_id,
        "created_at": datetime.now().isoformat(),
        "config": {
            "source": source,
            "concepts": concepts,
            "target_model": target_model,
            "topk_per_concept": topk_per_concept
        },
        "status": "pending_review",
        "features": features_with_details,
        "rejected_by_agent": rejected_features,
        "overall_strategy": overall_strategy
    }

    # Check if human review is enabled
    config = load_pipeline_config()
    human_review_enabled = config.get("feature_selection", {}).get("human_review_enabled", True)

    if human_review_enabled:
        # Save to pending features for review
        pending = load_pending_features()
        pending["batches"].append(batch)
        save_pending_features(pending)

        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "features_selected": len(features_with_details),
            "features_rejected": len(rejected_features),
            "concepts_used": concepts,
            "destination": "pending_review",
            "redirect": "/feature-review"
        })
    else:
        # Add features directly to feature_set.json
        feature_set = load_feature_set()
        added_count = 0

        for feature in features_with_details:
            # Check if feature already exists
            already_exists = any(
                f["layer"] == feature["layer"] and str(f["index"]) == str(feature["index"])
                for f in feature_set["features"]
            )

            if not already_exists:
                new_feature = {
                    "modelId": feature_set.get("model", target_model),
                    "layer": feature["layer"],
                    "index": feature["index"],
                    "strength": feature["recommended_strength"],
                    "description": feature["description"],
                    "enabled": True
                }
                feature_set["features"].append(new_feature)
                added_count += 1

        save_feature_set(feature_set)

        return jsonify({
            "success": True,
            "batch_id": batch_id,
            "features_selected": len(features_with_details),
            "features_rejected": len(rejected_features),
            "features_added": added_count,
            "concepts_used": concepts,
            "destination": "feature_set"
        })


@app.route("/api/pending-features/<batch_id>")
def api_get_pending_feature_batch(batch_id):
    """Get a specific pending feature batch."""
    pending = load_pending_features()
    for batch in pending.get("batches", []):
        if batch["id"] == batch_id:
            return jsonify(batch)
    return jsonify({"error": "Batch not found"}), 404


@app.route("/api/pending-features/<batch_id>/feature/<feature_id>/approve", methods=["POST"])
def api_approve_pending_feature(batch_id, feature_id):
    """Approve a feature and add to feature_set.json."""
    try:
        data = request.json or {}
        pending = load_pending_features()
        feature_set = load_feature_set()

        for batch in pending.get("batches", []):
            if batch["id"] == batch_id:
                for feature in batch.get("features", []):
                    if feature["id"] == feature_id:
                        # Check if already decided
                        if feature.get("review_status") == "approved":
                            return jsonify({"error": "Feature already approved"}), 400
                        if feature.get("review_status") == "rejected":
                            return jsonify({"error": "Feature already rejected"}), 400

                        # Mark as approved
                        feature["review_status"] = "approved"
                        feature["decision_at"] = datetime.now().isoformat()

                        # Allow overriding strength
                        if "strength" in data:
                            feature["recommended_strength"] = data["strength"]

                        # Add to feature set
                        already_exists = any(
                            f["layer"] == feature["layer"] and str(f["index"]) == str(feature["index"])
                            for f in feature_set["features"]
                        )

                        if not already_exists:
                            new_feature = {
                                "modelId": feature_set.get("model", batch["config"].get("target_model", "llama3.1-8b-it")),
                                "layer": feature["layer"],
                                "index": int(feature["index"]),
                                "strength": feature["recommended_strength"],
                                "description": feature["description"],
                                "enabled": True
                            }
                            feature_set["features"].append(new_feature)
                            save_feature_set(feature_set)

                        save_pending_features(pending)

                        return jsonify({
                            "success": True,
                            "feature_id": feature_id,
                            "added_to_feature_set": not already_exists
                        })

                return jsonify({"error": "Feature not found in batch"}), 404

        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to approve feature: {str(e)}"}), 500


@app.route("/api/pending-features/<batch_id>/feature/<feature_id>/reject", methods=["POST"])
def api_reject_pending_feature(batch_id, feature_id):
    """Reject a feature."""
    try:
        data = request.json or {}
        pending = load_pending_features()

        for batch in pending.get("batches", []):
            if batch["id"] == batch_id:
                for feature in batch.get("features", []):
                    if feature["id"] == feature_id:
                        # Check if already decided
                        if feature.get("review_status") == "approved":
                            return jsonify({"error": "Feature already approved"}), 400
                        if feature.get("review_status") == "rejected":
                            return jsonify({"error": "Feature already rejected"}), 400

                        # Mark as rejected
                        feature["review_status"] = "rejected"
                        feature["decision_at"] = datetime.now().isoformat()
                        feature["rejection_reason"] = data.get("reason", "Rejected by reviewer")

                        save_pending_features(pending)

                        return jsonify({
                            "success": True,
                            "feature_id": feature_id
                        })

                return jsonify({"error": "Feature not found in batch"}), 404

        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to reject feature: {str(e)}"}), 500


@app.route("/api/pending-features/<batch_id>/approve-all", methods=["POST"])
def api_approve_all_pending_features(batch_id):
    """Approve all pending features in a batch."""
    try:
        pending = load_pending_features()
        feature_set = load_feature_set()

        for batch in pending.get("batches", []):
            if batch["id"] == batch_id:
                approved_count = 0

                for feature in batch.get("features", []):
                    # Only approve features that haven't been decided yet
                    if feature.get("review_status") == "pending":
                        feature["review_status"] = "approved"
                        feature["decision_at"] = datetime.now().isoformat()

                        # Add to feature set
                        already_exists = any(
                            f["layer"] == feature["layer"] and str(f["index"]) == str(feature["index"])
                            for f in feature_set["features"]
                        )

                        if not already_exists:
                            new_feature = {
                                "modelId": feature_set.get("model", batch["config"].get("target_model", "llama3.1-8b-it")),
                                "layer": feature["layer"],
                                "index": int(feature["index"]),
                                "strength": feature["recommended_strength"],
                                "description": feature["description"],
                                "enabled": True
                            }
                            feature_set["features"].append(new_feature)

                        approved_count += 1

                save_pending_features(pending)
                save_feature_set(feature_set)

                return jsonify({
                    "success": True,
                    "approved_count": approved_count
                })

        return jsonify({"error": "Batch not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to approve all features: {str(e)}"}), 500


@app.route("/api/pending-features/<batch_id>", methods=["DELETE"])
def api_delete_pending_feature_batch(batch_id):
    """Delete a pending feature batch."""
    pending = load_pending_features()
    original_count = len(pending.get("batches", []))
    pending["batches"] = [b for b in pending.get("batches", []) if b["id"] != batch_id]

    if len(pending["batches"]) < original_count:
        save_pending_features(pending)
        return jsonify({"success": True})

    return jsonify({"error": "Batch not found"}), 404


# ============================================================
# Training Datasets API
# ============================================================

@app.route("/training-data")
def training_data_page():
    """Training Data page."""
    datasets_data = load_training_datasets()
    training_data = load_training_data()
    return render_template(
        "training_data.html",
        datasets=datasets_data.get("datasets", []),
        default_dataset_id=datasets_data.get("default_dataset_id"),
        data_points=training_data.get("data_points", [])
    )


@app.route("/api/datasets")
def api_datasets_list():
    """List all datasets."""
    datasets_data = load_training_datasets()
    training_data = load_training_data()

    # Add data point counts
    for dataset in datasets_data["datasets"]:
        dataset["data_point_count"] = len([
            dp for dp in training_data["data_points"]
            if dp.get("dataset_id") == dataset["id"]
        ])

    return jsonify(datasets_data)


@app.route("/api/datasets", methods=["POST"])
def api_datasets_create():
    """Create a new dataset."""
    data = request.json
    datasets_data = load_training_datasets()

    dataset = {
        "id": generate_id("ds"),
        "name": data.get("name", "New Dataset"),
        "description": data.get("description", ""),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "version": 1,
        "locked": False
    }

    datasets_data["datasets"].append(dataset)

    # Set as default if first dataset
    if len(datasets_data["datasets"]) == 1:
        datasets_data["default_dataset_id"] = dataset["id"]

    save_training_datasets(datasets_data)

    return jsonify({"success": True, "dataset": dataset})


@app.route("/api/datasets/<dataset_id>")
def api_datasets_get(dataset_id):
    """Get a specific dataset."""
    datasets_data = load_training_datasets()
    for dataset in datasets_data["datasets"]:
        if dataset["id"] == dataset_id:
            return jsonify(dataset)
    return jsonify({"error": "Dataset not found"}), 404


@app.route("/api/datasets/<dataset_id>", methods=["PUT"])
def api_datasets_update(dataset_id):
    """Update a dataset."""
    data = request.json
    datasets_data = load_training_datasets()

    for i, dataset in enumerate(datasets_data["datasets"]):
        if dataset["id"] == dataset_id:
            for key in ["name", "description", "locked"]:
                if key in data:
                    dataset[key] = data[key]
            dataset["updated_at"] = datetime.now().isoformat()
            datasets_data["datasets"][i] = dataset
            save_training_datasets(datasets_data)
            return jsonify({"success": True, "dataset": dataset})

    return jsonify({"error": "Dataset not found"}), 404


@app.route("/api/datasets/<dataset_id>", methods=["DELETE"])
def api_datasets_delete(dataset_id):
    """Delete a dataset and its data points."""
    datasets_data = load_training_datasets()
    training_data = load_training_data()

    original_count = len(datasets_data["datasets"])
    datasets_data["datasets"] = [d for d in datasets_data["datasets"] if d["id"] != dataset_id]

    if len(datasets_data["datasets"]) < original_count:
        # Also remove data points
        training_data["data_points"] = [
            dp for dp in training_data["data_points"]
            if dp.get("dataset_id") != dataset_id
        ]
        save_training_datasets(datasets_data)
        save_training_data(training_data)
        return jsonify({"success": True})

    return jsonify({"error": "Dataset not found"}), 404


# ============================================================
# Training Data Points API
# ============================================================

@app.route("/api/training-data")
def api_training_data_list():
    """List training data points, optionally filtered by dataset."""
    training_data = load_training_data()
    dataset_id = request.args.get("dataset_id")

    data_points = training_data.get("data_points", [])
    if dataset_id:
        data_points = [dp for dp in data_points if dp.get("dataset_id") == dataset_id]

    return jsonify(data_points)


@app.route("/api/training-data", methods=["POST"])
def api_training_data_add():
    """Add a training data point."""
    data = request.json
    training_data = load_training_data()

    data_point = {
        "id": generate_id("dp"),
        "dataset_id": data.get("dataset_id"),
        "scenario": data.get("scenario", {}),
        "default_response": data.get("default_response", ""),
        "steered_response": data.get("steered_response", ""),
        "feature": data.get("feature", {}),
        "strength": data.get("strength", 0),
        "source": data.get("source", {}),
        "created_at": datetime.now().isoformat(),
        "notes": data.get("notes", "")
    }

    training_data["data_points"].append(data_point)
    save_training_data(training_data)

    # Update dataset timestamp
    datasets_data = load_training_datasets()
    for dataset in datasets_data["datasets"]:
        if dataset["id"] == data_point["dataset_id"]:
            dataset["updated_at"] = datetime.now().isoformat()
            save_training_datasets(datasets_data)
            break

    return jsonify({"success": True, "data_point": data_point})


@app.route("/api/training-data/<data_point_id>")
def api_training_data_get(data_point_id):
    """Get a specific data point."""
    training_data = load_training_data()
    for dp in training_data["data_points"]:
        if dp["id"] == data_point_id:
            return jsonify(dp)
    return jsonify({"error": "Data point not found"}), 404


@app.route("/api/training-data/<data_point_id>", methods=["PUT"])
def api_training_data_update(data_point_id):
    """Update a data point."""
    data = request.json
    training_data = load_training_data()

    for i, dp in enumerate(training_data["data_points"]):
        if dp["id"] == data_point_id:
            for key in ["notes"]:
                if key in data:
                    dp[key] = data[key]
            training_data["data_points"][i] = dp
            save_training_data(training_data)
            return jsonify({"success": True, "data_point": dp})

    return jsonify({"error": "Data point not found"}), 404


@app.route("/api/training-data/<data_point_id>", methods=["DELETE"])
def api_training_data_delete(data_point_id):
    """Delete a data point."""
    training_data = load_training_data()
    original_count = len(training_data["data_points"])
    training_data["data_points"] = [
        dp for dp in training_data["data_points"]
        if dp["id"] != data_point_id
    ]

    if len(training_data["data_points"]) < original_count:
        save_training_data(training_data)
        return jsonify({"success": True})

    return jsonify({"error": "Data point not found"}), 404


@app.route("/api/training-data/<data_point_id>/move", methods=["POST"])
def api_training_data_move(data_point_id):
    """Move a data point to a different dataset."""
    data = request.json
    target_dataset_id = data.get("target_dataset_id")

    if not target_dataset_id:
        return jsonify({"error": "target_dataset_id is required"}), 400

    training_data = load_training_data()
    datasets_data = load_training_datasets()

    # Verify target dataset exists
    if not any(d["id"] == target_dataset_id for d in datasets_data["datasets"]):
        return jsonify({"error": "Target dataset not found"}), 404

    for i, dp in enumerate(training_data["data_points"]):
        if dp["id"] == data_point_id:
            old_dataset_id = dp.get("dataset_id")
            dp["dataset_id"] = target_dataset_id
            training_data["data_points"][i] = dp
            save_training_data(training_data)

            # Update both dataset timestamps
            for dataset in datasets_data["datasets"]:
                if dataset["id"] in [old_dataset_id, target_dataset_id]:
                    dataset["updated_at"] = datetime.now().isoformat()
            save_training_datasets(datasets_data)

            return jsonify({"success": True, "data_point": dp})

    return jsonify({"error": "Data point not found"}), 404


@app.route("/api/datasets/<dataset_id>/export")
def api_datasets_export(dataset_id):
    """Export a dataset as JSON."""
    datasets_data = load_training_datasets()
    training_data = load_training_data()

    dataset = None
    for d in datasets_data["datasets"]:
        if d["id"] == dataset_id:
            dataset = d
            break

    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404

    data_points = [
        dp for dp in training_data["data_points"]
        if dp.get("dataset_id") == dataset_id
    ]

    export = {
        "dataset": dataset,
        "data_points": data_points,
        "exported_at": datetime.now().isoformat()
    }

    return jsonify(export)


# ============== STEERING EXPERIMENTS ==============

@app.route("/steering-experiments")
def steering_experiments_page():
    """Steering experiments page - run features against scenarios."""
    approved_features = get_approved_features()

    bank = load_scenario_bank()
    enabled_scenarios = [s for s in bank.get("scenarios", []) if s.get("enabled", True)]

    experiments_data = load_steering_experiments()
    runs = experiments_data.get("experiment_runs", [])

    config = load_pipeline_config()
    steering_params = config.get("steering_params", {})
    test_strengths = steering_params.get("test_strengths", [2, 5, 10])

    # Compute concept overlap
    concept_features = {}
    for f in approved_features:
        c = f.get("concept", "")
        if c:
            concept_features.setdefault(c, []).append(f["id"])

    concept_scenarios = {}
    for s in enabled_scenarios:
        for c in s.get("candidate_concepts", []):
            concept_scenarios.setdefault(c, []).append(s.get("id", ""))

    shared_concepts = sorted(set(concept_features.keys()) & set(concept_scenarios.keys()))

    return render_template("steering_experiments.html",
        approved_features=approved_features,
        enabled_scenarios=enabled_scenarios,
        runs=runs,
        test_strengths=test_strengths,
        shared_concepts=shared_concepts,
        concept_features=concept_features,
        concept_scenarios=concept_scenarios,
        target_model=steering_params.get("target_model", "llama3.1-8b-it")
    )


@app.route("/api/steering-experiments/start", methods=["POST"])
def api_start_steering_experiment():
    data = request.json or {}

    feature_ids = data.get("feature_ids", "all")
    scenario_ids = data.get("scenario_ids", "all")
    target_model = data.get("target_model", "llama3.1-8b-it")
    test_strengths = data.get("test_strengths")

    if not test_strengths:
        config = load_pipeline_config()
        test_strengths = config.get("steering_params", {}).get("test_strengths", [2, 5, 10])

    # Collect features
    all_approved = get_approved_features()
    if feature_ids == "all":
        selected_features = all_approved
    else:
        selected_features = [f for f in all_approved if f["id"] in feature_ids]

    if not selected_features:
        return jsonify({"error": "No approved features found"}), 400

    # Collect scenarios
    bank = load_scenario_bank()
    all_scenarios = [s for s in bank.get("scenarios", []) if s.get("enabled", True)]
    if scenario_ids == "all":
        selected_scenarios = all_scenarios
    else:
        selected_scenarios = [s for s in all_scenarios if s.get("id") in scenario_ids]

    if not selected_scenarios:
        return jsonify({"error": "No enabled scenarios found"}), 400

    # Build feature_directions from feature metadata
    feature_directions = {}
    for f in selected_features:
        key = (f["layer"], int(f["index"]))
        feature_directions[key] = {
            "direction": f.get("direction", "positive"),
            "reasoning": f.get("direction_reasoning", "")
        }

    # Create run record
    run_id = generate_id("stexp")
    run = {
        "id": run_id,
        "created_at": datetime.now().isoformat(),
        "status": "running",
        "config": {
            "target_model": target_model,
            "test_strengths": test_strengths,
            "feature_count": len(selected_features),
            "scenario_count": len(selected_scenarios)
        },
        "progress": {"step": "starting", "current": 0, "total": 0},
        "error": None,
        "completed_at": None,
        "results": [],
        "evaluation_summary": None
    }

    experiments_data = load_steering_experiments()
    experiments_data["experiment_runs"].append(run)
    save_steering_experiments(experiments_data)

    # Background thread function
    def _run_thread():
        from pipeline import run_steering_experiments_standalone
        try:
            def progress_cb(current, total, **kwargs):
                experiments_data = load_steering_experiments()
                for r in experiments_data["experiment_runs"]:
                    if r["id"] == run_id:
                        r["progress"] = {
                            "step": kwargs.get("step", "steering"),
                            "current": current,
                            "total": total,
                            "current_feature": kwargs.get("current_feature", ""),
                            "current_strength": kwargs.get("current_strength", 0),
                            "current_scenario": kwargs.get("current_scenario", "")
                        }
                        break
                save_steering_experiments(experiments_data)

            def cancel_check():
                return _active_experiments.get(run_id, {}).get("cancelled", False)

            results = run_steering_experiments_standalone(
                scenarios=selected_scenarios,
                features=selected_features,
                target_model=target_model,
                test_strengths=test_strengths,
                feature_directions=feature_directions,
                progress_callback=progress_cb,
                cancel_check=cancel_check
            )

            experiments_data = load_steering_experiments()
            for r in experiments_data["experiment_runs"]:
                if r["id"] == run_id:
                    r["results"] = results
                    r["status"] = "cancelled" if cancel_check() else "completed"
                    r["completed_at"] = datetime.now().isoformat()
                    break
            save_steering_experiments(experiments_data)

        except Exception as e:
            experiments_data = load_steering_experiments()
            for r in experiments_data["experiment_runs"]:
                if r["id"] == run_id:
                    r["status"] = "failed"
                    r["error"] = str(e)
                    r["completed_at"] = datetime.now().isoformat()
                    break
            save_steering_experiments(experiments_data)
        finally:
            _active_experiments.pop(run_id, None)

    thread = threading.Thread(target=_run_thread, daemon=True)
    _active_experiments[run_id] = {"thread": thread, "cancelled": False}
    thread.start()

    return jsonify({"success": True, "run_id": run_id})


@app.route("/api/steering-experiments/<run_id>/status")
def api_steering_experiment_status(run_id):
    experiments_data = load_steering_experiments()
    for r in experiments_data["experiment_runs"]:
        if r["id"] == run_id:
            return jsonify({
                "status": r["status"],
                "progress": r.get("progress", {}),
                "error": r.get("error")
            })
    return jsonify({"error": "Run not found"}), 404


@app.route("/api/steering-experiments/<run_id>/cancel", methods=["POST"])
def api_cancel_steering_experiment(run_id):
    if run_id in _active_experiments:
        _active_experiments[run_id]["cancelled"] = True
        return jsonify({"success": True})
    return jsonify({"error": "No active experiment with this ID"}), 404


@app.route("/api/steering-experiments/<run_id>")
def api_get_steering_experiment(run_id):
    experiments_data = load_steering_experiments()
    for r in experiments_data["experiment_runs"]:
        if r["id"] == run_id:
            return jsonify(r)
    return jsonify({"error": "Run not found"}), 404


@app.route("/api/steering-experiments/<run_id>", methods=["DELETE"])
def api_delete_steering_experiment(run_id):
    experiments_data = load_steering_experiments()
    experiments_data["experiment_runs"] = [r for r in experiments_data["experiment_runs"] if r["id"] != run_id]
    save_steering_experiments(experiments_data)
    return jsonify({"success": True})


@app.route("/api/steering-experiments/<run_id>/evaluate", methods=["POST"])
def api_evaluate_steering_experiment(run_id):
    data = request.json or {}
    high_quality_mode = data.get("high_quality_mode", False)

    experiments_data = load_steering_experiments()
    run = None
    for r in experiments_data["experiment_runs"]:
        if r["id"] == run_id:
            run = r
            break

    if not run:
        return jsonify({"error": "Run not found"}), 404

    if run["status"] != "completed":
        return jsonify({"error": "Run not completed yet"}), 400

    # Run evaluation in background
    def _eval_thread():
        from agents import EvaluationJudgeAgent, DEFAULT_MODEL, HIGH_QUALITY_MODEL

        model = HIGH_QUALITY_MODEL if high_quality_mode else DEFAULT_MODEL
        judge = EvaluationJudgeAgent(model_name=model)

        total_evals = 0
        for feat_result in run["results"]:
            for strength, scenario_results in feat_result.get("strength_results", {}).items():
                for sr in scenario_results:
                    if sr.get("evaluation") is None and sr.get("error") is None:
                        total_evals += 1

        current_eval = 0
        successes = 0
        partials = 0
        failures = 0
        inconclusives = 0

        experiments_data = load_steering_experiments()
        for r in experiments_data["experiment_runs"]:
            if r["id"] != run_id:
                continue

            for feat_result in r["results"]:
                feature_info = feat_result["feature"]
                for strength_key, scenario_results in feat_result.get("strength_results", {}).items():
                    for sr in scenario_results:
                        if sr.get("evaluation") is not None or sr.get("error"):
                            continue

                        current_eval += 1
                        r["progress"] = {
                            "step": "evaluating",
                            "current": current_eval,
                            "total": total_evals,
                            "current_feature": f"Layer {feature_info.get('layer', '')}",
                            "current_strength": strength_key,
                            "current_scenario": sr.get("scenario_name", "")
                        }
                        save_steering_experiments(experiments_data)

                        try:
                            eval_result = judge.evaluate(
                                scenario={"name": sr["scenario_name"], "messages": []},
                                default_response=sr["default"],
                                steered_response=sr["steered"],
                                features_applied=[{"layer": feature_info["layer"], "index": feature_info["index"], "strength": int(strength_key)}]
                            )
                            sr["evaluation"] = eval_result

                            classification = eval_result.get("classification", "")
                            if classification == "success":
                                successes += 1
                            elif classification == "partial":
                                partials += 1
                            elif classification == "failure":
                                failures += 1
                            else:
                                inconclusives += 1
                        except Exception as e:
                            sr["evaluation"] = {"classification": "error", "error": str(e)}
                            failures += 1

            r["evaluation_summary"] = {
                "successes": successes,
                "partials": partials,
                "failures": failures,
                "inconclusives": inconclusives,
                "total": total_evals
            }
            r["progress"] = {"step": "complete", "current": total_evals, "total": total_evals}
            break

        save_steering_experiments(experiments_data)
        _active_experiments.pop(f"eval_{run_id}", None)

    thread = threading.Thread(target=_eval_thread, daemon=True)
    _active_experiments[f"eval_{run_id}"] = {"thread": thread, "cancelled": False}
    thread.start()

    return jsonify({"success": True, "message": "Evaluation started"})


@app.route("/api/steering-experiments/<run_id>/export", methods=["POST"])
def api_export_steering_results(run_id):
    data = request.json or {}
    min_classification = data.get("min_classification", "success")  # "success" or "partial"

    experiments_data = load_steering_experiments()
    run = None
    for r in experiments_data["experiment_runs"]:
        if r["id"] == run_id:
            run = r
            break

    if not run:
        return jsonify({"error": "Run not found"}), 404

    acceptable = {"success"}
    if min_classification == "partial":
        acceptable.add("partial")

    exported = []
    for feat_result in run.get("results", []):
        feature = feat_result["feature"]
        for strength_key, scenario_results in feat_result.get("strength_results", {}).items():
            for sr in scenario_results:
                eval_data = sr.get("evaluation")
                if eval_data and eval_data.get("classification") in acceptable:
                    exported.append({
                        "scenario_name": sr["scenario_name"],
                        "scenario_id": sr.get("scenario_id", ""),
                        "default": sr["default"],
                        "steered": sr["steered"],
                        "feature": feature,
                        "strength": int(strength_key),
                        "classification": eval_data["classification"],
                        "confidence": eval_data.get("confidence", 0),
                        "exported_from": run_id,
                        "exported_at": datetime.now().isoformat()
                    })

    if not exported:
        return jsonify({"error": "No results matching criteria to export", "exported": 0}), 400

    # Append to training data file
    training_file = BASE_DIR / "training_data.json"
    if training_file.exists():
        with open(training_file, 'r') as f:
            training_data = json.load(f)
    else:
        training_data = {"items": []}

    training_data["items"].extend(exported)
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)

    return jsonify({"success": True, "exported": len(exported)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
