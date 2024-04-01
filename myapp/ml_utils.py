# ml_utils.py
'''
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from PIL import Image

def load_custom_model(model_path):
    """
    Load the custom ML model from the specified path.
    
    Args:
    - model_path (str): Path to the saved model file.
    
    Returns:
    - model: Loaded ML model object.
    """
    with open(model_path, 'rb') as file:
        model = load_model(file)
    return model

def preprocess_image(image_data):
    """
    Preprocess image data before passing it to the ML model.
    
    Args:
    - image_data (bytes): Image data retrieved from MongoDB.
    
    Returns:
    - processed_image: Preprocessed image data ready for prediction.
    """
    # Example preprocessing steps (you may need to adjust based on your model requirements)
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))  # Resize image to match model input shape
    image = np.array(image) / 255.0   # Normalize pixel values
    # Additional preprocessing steps can be added here
    
    return image

def make_prediction(model, image):
    """
    Make predictions using the loaded ML model.
    
    Args:
    - model: Loaded ML model object.
    - image: Preprocessed image data.
    
    Returns:
    - prediction: Prediction result from the ML model.
    """
    # Example: Assuming your model has a predict method
    prediction = model.predict(image)
    return prediction
'''