from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('cat_vs_dog_model.h5')

def predict_image(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))  # Match training size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return f"Dog (Confidence: {prediction[0][0]:.2f})"
    else:
        return f"Cat (Confidence: {1 - prediction[0][0]:.2f})"

# Example usage
print(predict_image("O:\Self imp\Python\Learning of Python\25 March\Cat_November_2010-1a.jpg", model))
