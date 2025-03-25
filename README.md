# Cat vs Dog Classifier - TensorFlow/Keras
```
Minimum recommended: 1000 images per class for good accuracy.

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/cat-dog-classifier.git
cd cat-dog-classifier
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the training script:
```bash
python main.py
```

## 🚀 Usage
### 1. Training the Model
```bash
python main.py
```
Automatically trains the model and saves it as `cat_dog_classifier.keras`.
Generates training plots (`training_history.png`).

### 2. Making Predictions
Use the `predict_image()` function:
```python
from tensorflow.keras.models import load_model

model = load_model('cat_dog_classifier.keras')
prediction = predict_image('test_image.jpg', model)
print(prediction)  # Output: "Cat (Confidence: 0.92)" or "Dog (Confidence: 0.89)"
```

## 📊 Training Results
After training, expect:
- **Training Accuracy**: ~90-95%
- **Validation Accuracy**: ~85-90%
- **Training Loss**: Decreases steadily
- **Validation Loss**: Should not increase (indicates no overfitting)

## 🔮 Prediction
### Example Prediction
```python
test_image = "dataset/cat/cat_sample.jpg"
print(predict_image(test_image, model))
```
**Output:**
```bash
"Cat (Confidence: 0.94)"
```
### Supported Image Formats
- `.jpg`, `.jpeg`, `.png`


## 📢 Credits
- Built with TensorFlow/Keras
- Dataset: Kaggle Dogs vs Cats

## 💡 Contributing
Feel free to:
- Open issues for bugs/feature requests.
- Submit pull requests for improvements.

Happy Classifying! 🐱🐶

## 🔗 Links
- [GitHub Repo](https://github.com/Omg018/cat-dog-classifier)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Guide](https://keras.io/)
