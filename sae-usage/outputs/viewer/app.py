#!/usr/bin/env python3
"""Flask Results Viewer for SAE Probe Data Generation."""

import os
import sys
import json
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for
import numpy as np
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / '.env')

from src.model_manager import ModelManager
from src.sae_manager import SAEManager
from src.neuronpedia_client import NeuronpediaClient
from src.llm_agent import LLMAgent
from src import prompts

app = Flask(__name__)

# Base directory for outputs (parent of viewer/)
OUTPUTS_DIR = Path(__file__).parent.parent
CONFIG_PATH = Path(__file__).parent.parent.parent / "config_test.yaml"

# Global components (initialized at startup)
model_manager = None
sae_manager = None
neuronpedia_client = None
llm_agent = None
config = None


def get_all_runs():
    """Get list of all run directories with their metadata."""
    runs = []

    for item in OUTPUTS_DIR.iterdir():
        if item.is_dir() and item.name != 'viewer':
            # Try to load config to get run info
            config_path = item / 'config.yaml'
            pairs_path = item / 'pairs.json'
            validation_path = item / 'validation_results.json'

            if pairs_path.exists():
                with open(pairs_path, 'r') as f:
                    pairs_data = json.load(f)

                validation_data = None
                if validation_path.exists():
                    with open(validation_path, 'r') as f:
                        validation_data = json.load(f)

                avg_quality = None
                if validation_data:
                    quality_scores = [v['quality_score'] for v in validation_data]
                    avg_quality = np.mean(quality_scores) if quality_scores else None

                runs.append({
                    'timestamp': item.name,
                    'num_pairs': len(pairs_data),
                    'avg_quality': avg_quality,
                    'has_validation': validation_data is not None
                })

    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    return runs


def load_run_data(timestamp):
    """Load all data for a specific run."""
    run_dir = OUTPUTS_DIR / timestamp

    # Load pairs
    with open(run_dir / 'pairs.json', 'r') as f:
        pairs = json.load(f)

    # Load validation results
    validation_results = None
    if (run_dir / 'validation_results.json').exists():
        with open(run_dir / 'validation_results.json', 'r') as f:
            validation_results = json.load(f)

    # Load config
    config = None
    if (run_dir / 'config.yaml').exists():
        import yaml
        with open(run_dir / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

    # Load activations
    activations_dir = run_dir / 'activations'
    activations = {}
    if activations_dir.exists():
        for npy_file in activations_dir.glob('*.npy'):
            key = npy_file.stem  # e.g., 'pair_0_positive'
            activations[key] = np.load(npy_file)

    return {
        'timestamp': timestamp,
        'pairs': pairs,
        'validation_results': validation_results,
        'config': config,
        'activations': activations
    }


@app.route('/')
def index():
    """Run browser page."""
    runs = get_all_runs()
    return render_template('index.html', runs=runs)


@app.route('/run/<timestamp>')
def view_run(timestamp):
    """View detailed results for a specific run."""
    data = load_run_data(timestamp)
    return render_template('run.html', **data)


@app.route('/api/activation/<timestamp>/<pair_id>/<label>')
def get_activation_data(timestamp, pair_id, label):
    """API endpoint to get activation data for heatmap."""
    run_dir = OUTPUTS_DIR / timestamp
    activations_dir = run_dir / 'activations'

    filename = f'pair_{pair_id}_{label}.npy'
    filepath = activations_dir / filename

    if not filepath.exists():
        return jsonify({'error': 'Activation file not found'}), 404

    activations = np.load(filepath)

    # Convert to list for JSON serialization
    # Shape: [num_tokens, hidden_dim]
    # Convert float16 to regular Python floats for JSON compatibility
    return jsonify({
        'shape': list(activations.shape),
        'num_tokens': activations.shape[0],
        'hidden_dim': activations.shape[1],
        # Send mean activation per token for visualization
        'token_means': [float(x) for x in activations.mean(axis=1)],
        # Send top-k dimensions per token
        'top_dims_per_token': [
            {
                'token_idx': i,
                'top_indices': activations[i].argsort()[-10:][::-1].tolist(),
                'top_values': [float(x) for x in sorted(activations[i], reverse=True)[:10]]
            }
            for i in range(min(50, activations.shape[0]))  # Limit to first 50 tokens
        ]
    })


@app.route('/playground')
def playground():
    """Render interactive playground page."""
    # Get default prompts from prompts module
    default_prompts = {
        'topic_system': prompts.TOPIC_GENERATION_SYSTEM,
        'topic_user': prompts.TOPIC_GENERATION_USER,
        'question_system': prompts.QUESTION_GENERATION_SYSTEM,
        'question_user': prompts.QUESTION_GENERATION_USER,
        'positive_response_system': prompts.POSITIVE_RESPONSE_GENERATION_SYSTEM,
        'positive_response_user': prompts.POSITIVE_RESPONSE_GENERATION_USER,
        'negative_response_system': prompts.NEGATIVE_RESPONSE_GENERATION_SYSTEM,
        'negative_response_user': prompts.NEGATIVE_RESPONSE_GENERATION_USER,
    }

    return render_template(
        'playground.html',
        config=config,
        default_prompts=default_prompts
    )


@app.route('/playground/generate_topic', methods=['POST'])
def generate_topic():
    """Generate a topic using LLM agent."""
    try:
        data = request.json
        concept = data.get('concept', config['concept']['positive_class'])

        topic = llm_agent.generate_topic(concept)

        return jsonify({'success': True, 'topic': topic})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/playground/generate_question', methods=['POST'])
def generate_question():
    """Generate a question using LLM agent."""
    try:
        data = request.json
        positive_class = data.get('positive_class', config['concept']['positive_class'])
        negative_class = data.get('negative_class', config['concept']['negative_class'])
        topic = data.get('topic')

        question = llm_agent.generate_question(
            positive_class=positive_class,
            negative_class=negative_class,
            topic=topic
        )

        return jsonify({'success': True, 'question': question})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/playground/generate_response', methods=['POST'])
def generate_response():
    """Generate a response using LLM agent."""
    try:
        data = request.json
        positive_class = data.get('positive_class', config['concept']['positive_class'])
        negative_class = data.get('negative_class', config['concept']['negative_class'])
        question = data['question']
        is_positive = data['is_positive']
        reference_response = data.get('reference_response')

        # Update agent parameters if provided
        temperature = data.get('temperature', llm_agent.temperature)
        max_tokens = data.get('max_tokens', llm_agent.max_tokens)

        # Temporarily update agent parameters
        old_temp = llm_agent.temperature
        old_max = llm_agent.max_tokens
        llm_agent.temperature = temperature
        llm_agent.max_tokens = max_tokens

        response = llm_agent.generate_response(
            positive_class=positive_class,
            negative_class=negative_class,
            question=question,
            is_positive=is_positive,
            reference_response=reference_response
        )

        # Restore parameters
        llm_agent.temperature = old_temp
        llm_agent.max_tokens = old_max

        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/playground/analyze', methods=['POST'])
def analyze_pair():
    """Analyze a contrastive pair with activations and SAE features."""
    try:
        data = request.json
        question = data['question']
        positive_response = data['positive_response']
        negative_response = data['negative_response']
        layer = data.get('layer', config['model']['layer'])

        # Format conversation texts
        positive_text = f"User: {question}\nAssistant: {positive_response}"
        negative_text = f"User: {question}\nAssistant: {negative_response}"

        # Extract activations
        print(f"Extracting activations from layer {layer}...")
        positive_activations = model_manager.get_conversation_activations(positive_text, layer)
        negative_activations = model_manager.get_conversation_activations(negative_text, layer)

        # Compute cosine similarities
        print("Computing cosine similarities...")
        positive_features = sae_manager.compute_cosine_similarities(
            positive_activations,
            top_k=20,
            aggregate=config['sae']['aggregation']
        )
        negative_features = sae_manager.compute_cosine_similarities(
            negative_activations,
            top_k=20,
            aggregate=config['sae']['aggregation']
        )

        # Compute contrastive direction
        print("Computing contrastive direction...")
        contrastive_results = sae_manager.compute_contrastive_cosine_similarities(
            positive_activations,
            negative_activations,
            top_k=20,
            aggregate=config['sae']['aggregation']
        )

        # Get labels for all features
        all_feature_indices = (
            [idx for idx, _ in positive_features] +
            [idx for idx, _ in negative_features] +
            [idx for idx, _ in contrastive_results['positive']] +
            [idx for idx, _ in contrastive_results['negative']]
        )
        labels = neuronpedia_client.get_feature_labels(all_feature_indices)

        # Format results
        def format_features(features):
            return [
                {
                    'index': idx,
                    'cosine_similarity': float(sim),
                    'label': labels.get(idx, f'Feature {idx}')
                }
                for idx, sim in features
            ]

        results = {
            'positive_features': format_features(positive_features),
            'negative_features': format_features(negative_features),
            'contrastive_positive': format_features(contrastive_results['positive']),
            'contrastive_negative': format_features(contrastive_results['negative']),
            'num_positive_tokens': len(positive_activations),
            'num_negative_tokens': len(negative_activations)
        }

        return jsonify({'success': True, 'results': results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/playground/search_labels', methods=['POST'])
def search_labels():
    """Search for features by label keywords and compute their cosine similarities."""
    try:
        data = request.json
        keywords = data['keywords']  # List of keywords to search
        positive_text = data['positive_text']
        negative_text = data['negative_text']
        layer = data.get('layer', config['model']['layer'])

        # Extract activations
        positive_activations = model_manager.get_conversation_activations(positive_text, layer)
        negative_activations = model_manager.get_conversation_activations(negative_text, layer)

        # Aggregate activations to single vectors
        positive_mean = positive_activations.mean(axis=0)
        negative_mean = negative_activations.mean(axis=0)

        # Normalize
        positive_norm = positive_mean / (np.linalg.norm(positive_mean) + 1e-8)
        negative_norm = negative_mean / (np.linalg.norm(negative_mean) + 1e-8)

        # Ensure SAE is loaded
        if sae_manager.decoder_weights is None:
            sae_manager.load_sae()

        # Get decoder weights (move to CPU for numpy operations)
        decoder = sae_manager.decoder_weights  # Shape: (131072, 4096)
        if hasattr(decoder, 'cpu'):
            decoder = decoder.cpu().numpy()
        decoder_norm = decoder / (np.linalg.norm(decoder, axis=1, keepdims=True) + 1e-8)

        # Search for features using Neuronpedia API instead of cached labels
        keyword_results = []
        for keyword in keywords:
            # Use Neuronpedia API semantic search (top 5 most relevant features)
            search_results = neuronpedia_client.search_features(
                query=keyword,
                top_k=5
            )

            if not search_results:
                keyword_results.append({
                    'keyword': keyword,
                    'num_matches': 0,
                    'num_used': 0,
                    'positive_cosine': None,
                    'negative_cosine': None,
                    'difference': None,
                    'top_features': []
                })
                continue

            # Extract indices from search results
            matching_indices = [feat['index'] for feat in search_results]

            # Compute cosine similarities for matching features and store details
            positive_cosines = []
            negative_cosines = []
            feature_details = []

            for i, idx in enumerate(matching_indices):
                feature_vec = decoder_norm[idx]
                pos_cos = float(np.dot(positive_norm, feature_vec))
                neg_cos = float(np.dot(negative_norm, feature_vec))
                positive_cosines.append(pos_cos)
                negative_cosines.append(neg_cos)

                # Store individual feature details
                feature_details.append({
                    'index': int(idx),
                    'label': search_results[i]['description'],
                    'positive_cosine': pos_cos,
                    'negative_cosine': neg_cos,
                    'difference': pos_cos - neg_cos
                })

            # Aggregate (mean)
            mean_pos = float(np.mean(positive_cosines))
            mean_neg = float(np.mean(negative_cosines))

            keyword_results.append({
                'keyword': keyword,
                'num_matches': len(matching_indices),
                'num_used': len(matching_indices),
                'positive_cosine': mean_pos,
                'negative_cosine': mean_neg,
                'difference': mean_pos - mean_neg,
                'max_positive': float(np.max(positive_cosines)),
                'max_negative': float(np.max(negative_cosines)),
                'top_features': feature_details
            })

        return jsonify({'success': True, 'keyword_results': keyword_results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/playground/save', methods=['POST'])
def save_pair():
    """Save a manually created pair to outputs/manual/."""
    try:
        data = request.json
        pair_data = data['pair']

        # Create manual directory if it doesn't exist
        manual_dir = OUTPUTS_DIR / 'manual'
        manual_dir.mkdir(exist_ok=True)

        # Load existing pairs or create new list
        pairs_file = manual_dir / 'pairs.json'
        if pairs_file.exists():
            with open(pairs_file, 'r') as f:
                pairs = json.load(f)
        else:
            pairs = []

        # Assign pair_id
        pair_id = len(pairs)
        pair_data['pair_id'] = pair_id
        pair_data['format_type'] = 'qa'

        # Save pair metadata
        pairs.append(pair_data)
        with open(pairs_file, 'w') as f:
            json.dump(pairs, f, indent=2)

        # Save activations if provided
        if 'positive_activations' in pair_data and 'negative_activations' in pair_data:
            activations_dir = manual_dir / 'activations'
            activations_dir.mkdir(exist_ok=True)

            # Note: Activations need to be passed from frontend, or we re-extract them here
            # For now, let's re-extract them
            question = pair_data['question']
            positive_response = pair_data['positive_response']
            negative_response = pair_data['negative_response']
            layer = pair_data.get('layer', config['model']['layer'])

            positive_text = f"User: {question}\nAssistant: {positive_response}"
            negative_text = f"User: {question}\nAssistant: {negative_response}"

            positive_acts = model_manager.get_conversation_activations(positive_text, layer)
            negative_acts = model_manager.get_conversation_activations(negative_text, layer)

            np.save(activations_dir / f'pair_{pair_id}_positive.npy', positive_acts)
            np.save(activations_dir / f'pair_{pair_id}_negative.npy', negative_acts)

        # Save config snapshot if first pair
        config_file = manual_dir / 'config.yaml'
        if not config_file.exists():
            with open(config_file, 'w') as f:
                yaml.dump(config, f)

        return jsonify({
            'success': True,
            'pair_id': pair_id,
            'message': f'Saved pair {pair_id} to outputs/manual/'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/delete/<timestamp>', methods=['POST'])
def delete_run(timestamp):
    """Delete a run directory."""
    run_dir = OUTPUTS_DIR / timestamp

    # Safety check: only delete directories that look like timestamps
    if not run_dir.exists() or not run_dir.is_dir():
        return jsonify({'error': 'Run not found'}), 404

    # Safety check: don't delete the viewer directory
    if timestamp == 'viewer':
        return jsonify({'error': 'Cannot delete viewer directory'}), 403

    try:
        shutil.rmtree(run_dir)
        return jsonify({'success': True, 'message': f'Deleted run {timestamp}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===================================
# Probe Builder Routes
# ===================================

@app.route('/probe-builder')
def probe_builder():
    """Render the semantic probe builder page."""
    return render_template('probe_builder.html')


@app.route('/probe-builder/search', methods=['POST'])
def search_features():
    """Search for features by keyword using Neuronpedia API."""
    try:
        data = request.json
        keyword = data['keyword']

        # Use Neuronpedia API semantic search instead of cached labels
        search_results = neuronpedia_client.search_features(
            query=keyword,
            top_k=100  # Get more results for manual selection
        )

        # Format for frontend
        # Note: search_results are already sorted by cosine_similarity (relevance) from Neuronpedia API
        # We preserve this order so most relevant features appear first
        matching_features = [
            {
                'id': int(feat['index']),
                'label': feat['description']
            }
            for feat in search_results
        ]

        return jsonify({
            'success': True,
            'features': matching_features,
            'count': len(matching_features)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/probe-builder/test', methods=['POST'])
def test_probe():
    """Test a conversation against the current probe."""
    try:
        data = request.json
        system_prompt = data.get('system_prompt', '').strip()
        user_message = data['user_message']
        assistant_response = data['assistant_response']
        selected_features_a = data.get('selected_features_a', [])  # List of feature IDs for Probe A
        selected_features_b = data.get('selected_features_b', [])  # List of feature IDs for Probe B
        test_mode = data.get('test_mode', 'probe_a')  # 'probe_a', 'probe_b', or 'difference'
        layer = data.get('layer', config['model']['layer'])

        if not selected_features_a and not selected_features_b:
            return jsonify({'success': False, 'error': 'No features selected'}), 400

        # Format as conversation (include system prompt if provided)
        if system_prompt:
            conversation = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant: {assistant_response}"
        else:
            conversation = f"User: {user_message}\n\nAssistant: {assistant_response}"

        # Extract activations
        activations = model_manager.get_conversation_activations(conversation, layer)

        # Aggregate to single vector and normalize
        activation_mean = activations.mean(axis=0)
        activation_norm = activation_mean / (np.linalg.norm(activation_mean) + 1e-8)

        # Ensure SAE is loaded
        if sae_manager.decoder_weights is None:
            sae_manager.load_sae()

        # Get decoder weights
        decoder = sae_manager.decoder_weights  # Shape: (131072, 4096)
        if hasattr(decoder, 'cpu'):
            decoder = decoder.cpu().numpy()

        # If both probes have features, return all three results
        if selected_features_a and selected_features_b:
            # Compute Probe A direction
            indices_a = [int(f) for f in selected_features_a]
            directions_a = decoder[indices_a]
            probe_a = directions_a.mean(axis=0)
            probe_a = probe_a / (np.linalg.norm(probe_a) + 1e-8)
            cosine_sim_a = float(np.dot(activation_norm, probe_a))

            # Compute Probe B direction
            indices_b = [int(f) for f in selected_features_b]
            directions_b = decoder[indices_b]
            probe_b = directions_b.mean(axis=0)
            probe_b = probe_b / (np.linalg.norm(probe_b) + 1e-8)
            cosine_sim_b = float(np.dot(activation_norm, probe_b))

            # Compute difference direction: (A - B) / ||A - B||
            probe_diff = probe_a - probe_b
            probe_diff = probe_diff / (np.linalg.norm(probe_diff) + 1e-8)
            cosine_sim_diff = float(np.dot(activation_norm, probe_diff))

            return jsonify({
                'success': True,
                'cosine_similarity_a': cosine_sim_a,
                'cosine_similarity_b': cosine_sim_b,
                'cosine_similarity_diff': cosine_sim_diff,
                'num_features_a': len(selected_features_a),
                'num_features_b': len(selected_features_b),
                'activation_magnitude': float(np.linalg.norm(activation_mean))
            })

        # Single probe mode (only A or only B)
        selected_features = selected_features_a if selected_features_a else selected_features_b
        selected_indices = [int(f) for f in selected_features]
        selected_directions = decoder[selected_indices]

        # Compute probe direction (mean of selected features, normalized)
        probe_direction = selected_directions.mean(axis=0)
        probe_direction = probe_direction / (np.linalg.norm(probe_direction) + 1e-8)

        # Compute cosine similarity
        cosine_sim = float(np.dot(activation_norm, probe_direction))

        # Compute activation magnitude
        activation_magnitude = float(np.linalg.norm(activation_mean))

        return jsonify({
            'success': True,
            'cosine_similarity': cosine_sim,
            'activation_magnitude': activation_magnitude,
            'num_features': len(selected_features)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/probe-builder/save', methods=['POST'])
def save_probe():
    """Save the current probe(s) to disk."""
    try:
        data = request.json
        probe_name = data['probe_name']
        selected_features_a = data.get('selected_features_a', [])
        selected_features_b = data.get('selected_features_b', [])
        description = data.get('description', '')
        save_format = data.get('save_format', 'individual')  # 'individual', 'package', or 'both'

        if not probe_name:
            return jsonify({'success': False, 'error': 'Probe name required'}), 400

        if not selected_features_a and not selected_features_b:
            return jsonify({'success': False, 'error': 'No features selected'}), 400

        # Create probe directory if it doesn't exist
        probes_dir = OUTPUTS_DIR.parent / 'probes'
        probes_dir.mkdir(exist_ok=True)

        # Get labels for selected features
        all_labels = neuronpedia_client.labels
        saved_files = []

        def get_features_with_labels(feature_ids):
            """Helper to get features with their labels."""
            return [
                {'id': int(fid), 'label': all_labels.get(int(fid), '')}
                for fid in feature_ids
            ]

        # Save individual probes
        if save_format in ['individual', 'both']:
            if selected_features_a:
                probe_data_a = {
                    'name': f"{probe_name} - Probe A",
                    'description': description,
                    'features': get_features_with_labels(selected_features_a),
                    'created_at': datetime.now().isoformat()
                }
                filename_a = f"{probe_name.replace(' ', '_').lower()}_probe_a.json"
                filepath_a = probes_dir / filename_a
                with open(filepath_a, 'w') as f:
                    json.dump(probe_data_a, f, indent=2)
                saved_files.append(str(filepath_a))

            if selected_features_b:
                probe_data_b = {
                    'name': f"{probe_name} - Probe B",
                    'description': description,
                    'features': get_features_with_labels(selected_features_b),
                    'created_at': datetime.now().isoformat()
                }
                filename_b = f"{probe_name.replace(' ', '_').lower()}_probe_b.json"
                filepath_b = probes_dir / filename_b
                with open(filepath_b, 'w') as f:
                    json.dump(probe_data_b, f, indent=2)
                saved_files.append(str(filepath_b))

        # Save differential package
        if save_format in ['package', 'both']:
            if selected_features_a and selected_features_b:
                package_data = {
                    'name': f"{probe_name} - Differential",
                    'description': description,
                    'type': 'differential',
                    'probe_a': {
                        'features': get_features_with_labels(selected_features_a)
                    },
                    'probe_b': {
                        'features': get_features_with_labels(selected_features_b)
                    },
                    'created_at': datetime.now().isoformat()
                }
                filename_diff = f"{probe_name.replace(' ', '_').lower()}_differential.json"
                filepath_diff = probes_dir / filename_diff
                with open(filepath_diff, 'w') as f:
                    json.dump(package_data, f, indent=2)
                saved_files.append(str(filepath_diff))
            elif save_format == 'package':
                return jsonify({'success': False, 'error': 'Differential package requires features in both probes'}), 400

        if not saved_files:
            return jsonify({'success': False, 'error': 'No probes to save'}), 400

        message = f"Saved {len(saved_files)} file(s): " + ", ".join([f.split('/')[-1] for f in saved_files])

        return jsonify({
            'success': True,
            'message': message,
            'files': saved_files
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ===================================
# Steering Vector Tester Routes
# ===================================

@app.route('/steering')
def steering_page():
    """Render steering vector tester page."""
    return render_template('steering.html', config=config)


@app.route('/steering/get_label', methods=['POST'])
def get_steering_label():
    """Get label for a feature index."""
    try:
        data = request.json
        feature_index = int(data['feature_index'])

        # Validate feature index
        if feature_index < 0 or feature_index >= 131072:
            return jsonify({'success': False, 'error': 'Feature index must be between 0 and 131071'}), 400

        # Get label from neuronpedia client
        label = neuronpedia_client.get_label(feature_index)

        return jsonify({
            'success': True,
            'label': label,
            'feature_index': feature_index
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/steering/generate', methods=['POST'])
def generate_steered():
    """Generate text with and without steering applied."""
    try:
        data = request.json
        feature_index = int(data['feature_index'])
        steering_strength = float(data['steering_strength'])
        prompt = data['prompt']
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 256)
        layer = data.get('layer', config['model']['layer'])

        # Validate inputs
        if feature_index < 0 or feature_index >= 131072:
            return jsonify({'success': False, 'error': 'Feature index must be between 0 and 131071'}), 400

        if not prompt.strip():
            return jsonify({'success': False, 'error': 'Prompt cannot be empty'}), 400

        # Format as messages
        messages = [{'role': 'user', 'content': prompt}]

        # Generate baseline (no steering)
        print(f"Generating baseline response...")
        baseline_text = model_manager.generate_response(
            messages=messages,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        # Ensure SAE is loaded to get feature direction
        if sae_manager.decoder_weights is None:
            print("Loading SAE decoder for steering...")
            sae_manager.load_sae()

        # Get feature direction from SAE decoder
        feature_direction = sae_manager.decoder_weights[feature_index].cpu().numpy()

        # Generate with steering applied
        print(f"Generating steered response (feature {feature_index}, strength {steering_strength})...")
        steered_text = model_manager.generate_with_steering(
            messages=messages,
            feature_direction=feature_direction,
            steering_strength=steering_strength,
            layer=layer,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        # Get feature label
        feature_label = neuronpedia_client.get_label(feature_index)

        return jsonify({
            'success': True,
            'baseline_text': baseline_text,
            'steered_text': steered_text,
            'feature_index': feature_index,
            'feature_label': feature_label,
            'steering_strength': steering_strength,
            'layer': layer
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def initialize_components():
    """Initialize global components for playground."""
    global model_manager, sae_manager, neuronpedia_client, llm_agent, config

    print("\n" + "="*60)
    print("INITIALIZING PLAYGROUND COMPONENTS")
    print("="*60)

    # Load config
    print("1. Loading configuration...")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize LLM Agent
    print("2. Initializing LLM Agent...")
    api_key = os.getenv("OPEN_ROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY not found in environment variables")

    llm_agent = LLMAgent(
        api_key=api_key,
        model=config['llm_agent']['model'],
        temperature=config['llm_agent']['temperature'],
        max_tokens=config['llm_agent']['max_tokens']
    )

    # Initialize Model Manager
    print("3. Initializing Model Manager (Llama 3.1 8B)...")
    model_manager = ModelManager(
        model_name=config['model']['name'],
        device=config['model']['device'],
        load_in_8bit=config['model']['load_in_8bit'],
        max_length=config['model']['max_length']
    )

    # Initialize SAE Manager
    print("4. Initializing SAE Manager...")
    sae_manager = SAEManager(
        repo_id=config['sae']['repo_id'],
        filename=config['sae']['filename'],
        cache_dir=config['sae']['cache_dir'],
        device="cpu"  # Start on CPU to avoid OOM, will lazy-load to GPU when needed
    )
    print("   SAE will lazy-load when first accessed")

    # Initialize Neuronpedia Client
    print("5. Initializing Neuronpedia Client...")
    api_key = os.getenv(config['neuronpedia']['api_key_env'])
    if not api_key:
        raise ValueError(f"{config['neuronpedia']['api_key_env']} not found in environment variables")

    neuronpedia_client = NeuronpediaClient(
        api_key=api_key,
        model_id=config['neuronpedia']['model_id'],
        layer_id=config['neuronpedia']['layer_id'],
        cache_dir=config['neuronpedia'].get('cache_dir')
    )

    print("="*60)
    print("ALL COMPONENTS INITIALIZED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Initialize components for playground
    initialize_components()

    print("\n" + "="*60)
    print("SAE Probe Data Results Viewer + Interactive Playground")
    print("="*60)
    print(f"Serving results from: {OUTPUTS_DIR}")
    print(f"Found {len(get_all_runs())} runs")
    print("\nAvailable pages:")
    print("  - Run Browser: http://localhost:5000")
    print("  - Interactive Playground: http://localhost:5000/playground")
    print("  - Probe Builder: http://localhost:5000/probe-builder")
    print("  - Steering Tester: http://localhost:5000/steering")
    print("="*60 + "\n")

    app.run(debug=False, port=5000)
