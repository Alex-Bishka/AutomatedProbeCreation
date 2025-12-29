import os
import re
import json
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
STARRED_RUNS_FILE = BASE_DIR / "starred_runs.json"
RUNS_DIR = BASE_DIR.parent / "runs"

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


def load_runs():
    """Load all past runs from the runs directory."""
    runs = []
    starred = load_starred_runs()
    if RUNS_DIR.exists():
        for run_file in sorted(RUNS_DIR.glob("*.json"), reverse=True):
            with open(run_file, "r") as f:
                run_data = json.load(f)
                run_data["filename"] = run_file.name
                run_data["starred"] = run_file.name in starred

                # Calculate score from votes
                results = run_data.get("results", [])
                total = len(results)
                good_votes = sum(1 for r in results if r.get("vote") == "good")
                bad_votes = sum(1 for r in results if r.get("vote") == "bad")
                voted = good_votes + bad_votes

                # Score is proportion of good votes (only counting voted scenarios)
                if voted > 0:
                    run_data["score"] = good_votes / total
                    run_data["score_display"] = f"{good_votes}/{total}"
                else:
                    run_data["score"] = None  # No votes yet
                    run_data["score_display"] = "—"

                run_data["good_votes"] = good_votes
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


def save_run(results, feature_set, model):
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
    return render_template("features.html", feature_set=feature_set, archived=archived)


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
        "strength": data.get("strength", 10),
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


@app.route("/scenarios")
def scenarios_page():
    """Scenarios viewing and editing page."""
    scenarios = load_scenarios()
    return render_template("scenarios.html", scenarios=scenarios)


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
    return render_template("run.html", feature_set=feature_set, scenarios=scenarios)


@app.route("/api/run", methods=["POST"])
def api_run_scenarios():
    """Run selected scenarios with current feature set."""
    data = request.json
    scenario_ids = data.get("scenarios", [])

    feature_set = load_feature_set()
    scenarios = load_scenarios()

    # Filter to selected scenarios
    selected = [s for s in scenarios if s["id"] in scenario_ids]

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
                "default": default,
                "steered": steered,
                "error": None
            })
        except Exception as e:
            results.append({
                "name": scenario["name"],
                "scenario": scenario["messages"],
                "default": None,
                "steered": None,
                "error": str(e)
            })

    # Save the run (only save enabled features that were actually used)
    filename = save_run(results, enabled_features, feature_set["model"])

    return jsonify({
        "success": True,
        "results": results,
        "filename": filename
    })


@app.route("/runs")
def runs_page():
    """Browse past runs page."""
    runs = load_runs()
    return render_template("runs.html", runs=runs)


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


@app.route("/runs/<filename>")
def run_detail_page(filename):
    """View a specific run."""
    filepath = RUNS_DIR / filename
    if filepath.exists():
        with open(filepath, "r") as f:
            run_data = json.load(f)
        return render_template("run_detail.html", run=run_data, filename=filename)
    return redirect(url_for("runs_page"))


@app.route("/api/runs/<filename>/vote", methods=["POST"])
def api_vote_scenario(filename):
    """Vote on a scenario result within a run."""
    data = request.json
    scenario_index = data.get("scenario_index")
    vote = data.get("vote")  # "good", "bad", or None to clear

    filepath = RUNS_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "Run not found"}), 404

    with open(filepath, "r") as f:
        run_data = json.load(f)

    if scenario_index < 0 or scenario_index >= len(run_data["results"]):
        return jsonify({"error": "Invalid scenario index"}), 400

    # Set the vote
    run_data["results"][scenario_index]["vote"] = vote

    # Calculate score
    total = len(run_data["results"])
    good_votes = sum(1 for r in run_data["results"] if r.get("vote") == "good")
    bad_votes = sum(1 for r in run_data["results"] if r.get("vote") == "bad")
    score = good_votes / total if total > 0 else 0

    with open(filepath, "w") as f:
        json.dump(run_data, f, indent=2)

    return jsonify({
        "success": True,
        "score": score,
        "good_votes": good_votes,
        "bad_votes": bad_votes,
        "total": total
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
