from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import os
from pathlib import Path

app = Flask(__name__, static_folder='.')
CORS(app)

# Paths
MODEL_PATH = 'trained_model/best_model.h5'
SCALER_PATH = 'scaler.pkl'

# Load alphabet
try:
    from signlang_actions import ALPHABET
    print(f"✓ Loaded alphabet: {len(ALPHABET)} letters")
except ImportError:
    ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    print("⚠ Using default alphabet A-Z")

# Global variables for lazy loading
_model = None
_scaler = None

def get_model():
    """Lazy load LSTM model"""
    global _model
    if _model is None:
        try:
            from keras.models import load_model
            print("Loading LSTM model...")
            _model = load_model(MODEL_PATH)
            print(f"✓ LSTM model loaded successfully")
        except Exception as e:
            print(f"❌ Could not load LSTM model: {e}")
            import traceback
            traceback.print_exc()
            _model = False
    return _model if _model is not False else None

def get_scaler():
    """Lazy load scaler"""
    global _scaler
    if _scaler is None:
        try:
            with open(SCALER_PATH, 'rb') as f:
                _scaler = pickle.load(f)
            print(f"✓ Scaler loaded (expects {_scaler.n_features_in_} features)")
        except Exception as e:
            print(f"⚠ Could not load scaler: {e}")
            _scaler = False
    return _scaler if _scaler is not False else None

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
        
        if not data or 'sequence' not in data:
            return jsonify({'error': 'No sequence provided'}), 400
        
        sequence = np.array(data['sequence'])
        
        # Lazy load model
        model = get_model()
        if model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Validate sequence shape
        expected_length = 30
        expected_features = 258
        
        if sequence.shape[0] != expected_length:
            return jsonify({
                'error': f'Sequence length must be {expected_length}, got {sequence.shape[0]}'
            }), 400
        
        if sequence.shape[1] != expected_features:
            return jsonify({
                'error': f'Feature count must be {expected_features}, got {sequence.shape[1]}'
            }), 400
        
        # Lazy load scaler
        scaler = get_scaler()
        
        # Normalize using scaler if available
        if scaler is not None:
            n_features = sequence.shape[1]
            sequence_reshaped = sequence.reshape(-1, n_features)
            sequence_normalized = scaler.transform(sequence_reshaped)
            sequence = sequence_normalized.reshape(expected_length, n_features)
        
        # Make prediction
        sequence_input = np.expand_dims(sequence, axis=0)
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

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = 'loaded' if (_model is not None and _model is not False) else 'not_loaded'
    scaler_status = 'loaded' if (_scaler is not None and _scaler is not False) else 'not_loaded'
    
    return jsonify({
        'status': 'healthy',
        'model': 'lstm',
        'model_status': model_status,
        'scaler_status': scaler_status,
        'alphabet_size': len(ALPHABET),
        'expected_features': 258
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Sign Language Recognition Backend Server")
    print("="*60)
    print(f"Model: LSTM (Memory Optimized)")
    print(f"Expected Features: 258 (Pose: 132 + Hands: 126)")
    print(f"Alphabet: {len(ALPHABET)} letters")
    print("="*60)
    print("\nStarting server...")
    print("Model will be loaded on first prediction request")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
