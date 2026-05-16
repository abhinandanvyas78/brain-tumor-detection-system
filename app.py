from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables for models
MODELS = {}
MODEL_COMPARISON = None
IMG_SIZE = 224

# Model information
MODEL_INFO = {
    'VGG16': {
        'name': 'VGG16 Transfer Learning',
        'description': 'Deep convolutional network with transfer learning',
        'accuracy': 0.8431,
        'file': 'final_vgg16_model.h5'
    },
    'Custom CNN': {
        'name': 'Custom CNN',
        'description': 'Custom built convolutional neural network',
        'accuracy': 0.6667,
        'file': 'final_custom_cnn_model.h5'
    },
    'ResNet50': {
        'name': 'ResNet50',
        'description': 'Residual network with 50 layers',
        'accuracy': 0.7843,
        'file': 'final_resnet50_model.h5'
    },
    'MobileNetV2': {
        'name': 'MobileNetV2',
        'description': 'Lightweight efficient neural network',
        'accuracy': 0.8039,
        'file': 'final_mobilenet_model.h5'
    }
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_models():
    """Load all trained models"""
    global MODELS, MODEL_COMPARISON
    
    try:
        # Load model comparison data
        comparison_path = 'models/model_comparison.csv'
        if os.path.exists(comparison_path):
            MODEL_COMPARISON = pd.read_csv(comparison_path)
            print("✓ Model comparison data loaded")
        
        # Load VGG16 model (best model)
        vgg16_path = 'models/final_vgg16_model.h5'
        if os.path.exists(vgg16_path):
            MODELS['VGG16'] = load_model(vgg16_path)
            print("✓ VGG16 model loaded")
        
        # Load other models if available
        for model_name, info in MODEL_INFO.items():
            if model_name != 'VGG16':
                model_path = f'models/{info["file"]}'
                if os.path.exists(model_path):
                    try:
                        MODELS[model_name] = load_model(model_path)
                        print(f"✓ {model_name} model loaded")
                    except:
                        print(f"⚠ Could not load {model_name}")
        
        print(f"\\nTotal models loaded: {len(MODELS)}")
        
    except Exception as e:
        print(f"Error loading models: {e}")

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        # Resize to model input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    except Exception as e:
        raise Exception(f"Error preprocessing image: {e}")

def predict_tumor(image_path, model_name='VGG16'):
    """Predict brain tumor from MRI image"""
    try:
        # Check if model exists
        if model_name not in MODELS:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Get model
        model = MODELS[model_name]
        
        # Make prediction
        prediction_prob = model.predict(processed_img, verbose=0)[0][0]
        
        # Determine result
        has_tumor = prediction_prob > 0.5
        confidence = prediction_prob if has_tumor else (1 - prediction_prob)
        
        result = {
            'model': model_name,
            'has_tumor': bool(has_tumor),
            'prediction': 'Tumor Detected' if has_tumor else 'No Tumor Detected',
            'confidence': float(confidence * 100),
            'probability': float(prediction_prob * 100),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Prediction error: {e}")

def generate_visualization(image_path, prediction_result):
    """Generate visualization for prediction result"""
    try:
        # Read original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Brain Tumor Detection Result', fontsize=16, fontweight='bold')
        
        # Plot original image
        axes[0].imshow(img)
        axes[0].set_title('Original MRI Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Plot prediction result
        result_text = prediction_result['prediction']
        confidence = prediction_result['confidence']
        color = 'red' if prediction_result['has_tumor'] else 'green'
        
        axes[1].text(0.5, 0.6, result_text, 
                    ha='center', va='center', 
                    fontsize=20, fontweight='bold', color=color)
        axes[1].text(0.5, 0.4, f'Confidence: {confidence:.2f}%',
                    ha='center', va='center',
                    fontsize=16, fontweight='bold')
        axes[1].text(0.5, 0.2, f'Model: {prediction_result["model"]}',
                    ha='center', va='center',
                    fontsize=12)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        
        # Save to base64
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return image_base64
        
    except Exception as e:
        print(f"Visualization error: {e}")
        return None

# Routes
@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors"""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', 
                         models=MODEL_INFO,
                         total_models=len(MODELS))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/predict')
def predict_page():
    """Prediction page"""
    return render_template('predict.html', models=MODEL_INFO)

@app.route('/visualization')
def visualization():
    """Visualization page"""
    comparison_data = None
    if MODEL_COMPARISON is not None:
        comparison_data = MODEL_COMPARISON.to_dict('records')
    
    return render_template('visualization.html', 
                         models=MODEL_INFO,
                         comparison_data=comparison_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Get selected model
        model_name = request.form.get('model', 'VGG16')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_tumor(filepath, model_name)
        
        # Generate visualization
        viz_base64 = generate_visualization(filepath, result)
        
        # Add file path to result
        result['image_url'] = f'/static/uploads/{filename}'
        result['visualization'] = viz_base64
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def api_models():
    """Get available models information"""
    return jsonify({
        'models': MODEL_INFO,
        'loaded': list(MODELS.keys()),
        'total': len(MODELS)
    })

@app.route('/api/comparison')
def api_comparison():
    """Get model comparison data"""
    if MODEL_COMPARISON is not None:
        return jsonify(MODEL_COMPARISON.to_dict('records'))
    else:
        return jsonify({'error': 'Comparison data not available'}), 404

@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    return render_template('index.html', 
                         models=MODEL_INFO,
                         total_models=len(MODELS)), 404

@app.errorhandler(500)
def server_error(e):
    """500 error handler"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("BRAIN TUMOR DETECTION - Flask Application")
    print("By CodeAj Marketplace")
    print("=" * 80)
    print("\\nLoading models...")
    
    # Load models
    load_models()
    
    print("\\n" + "=" * 80)
    print("Starting Flask server...")
    print("=" * 80)
    print("\\nAccess the application at: http://localhost:8080")
    print("Press CTRL+C to quit\\n")
    
    # Run app
    app.run(debug=True, port=5000)
