from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def load_cat_dog_model(model_path='cat_dog_classifier.keras'):
    """Safely load the trained model"""
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(img_path, model):
    """Make prediction on a single image"""
    try:
        # Convert Windows path if needed
        img_path = os.path.normpath(img_path)
        
        if not os.path.exists(img_path):
            return f"Error: File not found at {img_path}"
            
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
        
        if confidence > 0.5:
            return f"Dog (Confidence: {confidence:.2f})"
        else:
            return f"Cat (Confidence: {1 - confidence:.2f})"
            
    except Exception as e:
        return f"Prediction error: {e}"

# Example usage
if __name__ == "__main__":
    # Load model
    model = load_cat_dog_model()
    
    if model:
        # Test prediction - use raw string or double backslashes for Windows paths
        image_path = r"O:\Self imp\Python\Learning of Python\25 March\Cat_November_2010-1a.jpg"
        result = predict_image(image_path, model)
        print(f"\nPrediction Result: {result}")