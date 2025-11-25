from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from keras.models import load_model
import pickle
import os
from pathlib import Path

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for web requests

# Load models and configurations
print("Loading models...")

# Paths
LSTM_MODEL_PATH = 'trained_model/best_model.h5'
TRANSFORMER_MODEL_PATH = 'trained_model_transformer/best_model_transformer.h5'
SCALER_PATH = 'scaler.pkl'

# Load models
try:
    lstm_model = load_model(LSTM_MODEL_PATH)
    print(f"✓ LSTM model loaded from {LSTM_MODEL_PATH}")
except Exception as e:
    print(f"⚠ Could not load LSTM model: {e}")
    lstm_model = None

try:
    transformer_model = load_model(TRANSFORMER_MODEL_PATH)
    print(f"✓ Transformer model loaded from {TRANSFORMER_MODEL_PATH}")
except Exception as e:
    print(f"⚠ Could not load Transformer model: {e}")
    transformer_model = None

# Load scaler
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {SCALER_PATH}")
    print(f"✓ Scaler expects {scaler.n_features_in_} features")
except Exception as e:
    print(f"⚠ Could not load scaler: {e}")
    scaler = None

# Load alphabet from signlang_actions.py
try:
    from signlang_actions import ALPHABET
    print(f"✓ Loaded alphabet: {len(ALPHABET)} letters")
except ImportError:
    ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    print("⚠ Using default alphabet A-Z")

# Model registry
models = {
    'lstm': lstm_model,
    'transformer': transformer_model
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data or 'sequence' not in data:
            return jsonify({'error': 'No sequence provided'}), 400
        
        sequence = np.array(data['sequence'])
        model_type = data.get('model', 'lstm')
        
        # Get the model
        model = models.get(model_type)
        if model is None:
            return jsonify({'error': f'Model {model_type} not loaded'}), 400
        
        # Validate sequence shape
        expected_length = 30
        expected_features = 258  # pose (132) + left_hand (63) + right_hand (63)
        
        if sequence.shape[0] != expected_length:
            return jsonify({
                'error': f'Sequence length must be {expected_length}, got {sequence.shape[0]}'
            }), 400
        
        if sequence.shape[1] != expected_features:
            return jsonify({
                'error': f'Feature count must be {expected_features}, got {sequence.shape[1]}. '
                        f'Ensure both pose and hand landmarks are being extracted.'
            }), 400
        
        print(f"Received sequence shape: {sequence.shape} ✓")
        
        # Normalize using scaler if available
        if scaler is not None:
            n_features = sequence.shape[1]
            sequence_reshaped = sequence.reshape(-1, n_features)
            sequence_normalized = scaler.transform(sequence_reshaped)
            sequence = sequence_normalized.reshape(expected_length, n_features)
        
        # Make prediction
        sequence_input = np.expand_dims(sequence, axis=0)  # Add batch dimension
        predictions = model.predict(sequence_input, verbose=0)[0]
        
        # Get predicted class
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])
        
        # Get predicted letter
        if predicted_index < len(ALPHABET):
            predicted_letter = ALPHABET[predicted_index]
        else:
            predicted_letter = '?'
        
        return jsonify({
            'prediction': predicted_letter,
            'confidence': confidence,
            'all_probabilities': predictions.tolist(),
            'predicted_index': predicted_index
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    available_models = []
    
    for model_name, model in models.items():
        if model is not None:
            available_models.append({
                'name': model_name,
                'status': 'loaded',
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape)
            })
        else:
            available_models.append({
                'name': model_name,
                'status': 'not_loaded'
            })
    
    return jsonify({
        'models': available_models,
        'alphabet': ALPHABET,
        'scaler_loaded': scaler is not None,
        'expected_features': 258
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'lstm_loaded': lstm_model is not None,
        'transformer_loaded': transformer_model is not None,
        'scaler_loaded': scaler is not None,
        'alphabet_size': len(ALPHABET),
        'expected_features': 258
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Sign Language Recognition Backend Server")
    print("="*60)
    print(f"LSTM Model: {'✓ Loaded' if lstm_model else '✗ Not loaded'}")
    print(f"Transformer Model: {'✓ Loaded' if transformer_model else '✗ Not loaded'}")
    print(f"Scaler: {'✓ Loaded' if scaler else '✗ Not loaded'}")
    print(f"Expected Features: 258 (Pose: 132 + Hands: 126)")
    print(f"Alphabet: {len(ALPHABET)} letters")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
