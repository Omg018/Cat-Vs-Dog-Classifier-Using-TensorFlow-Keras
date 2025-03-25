import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import conv2d, Maxpolining2d, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt

tf.random.set_seed(42)

IMG_SIZE = (150,150)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_PATH = 'test_set'

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset directory '{DATASET_PATH}' not found")

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

model = Sequential([
    Conv2d(32,(3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3) ),
    BatchNormalization(),
    Maxpolining2d(2,2),
    Dropout(0.25),

    Conv2d(64,(3,3), activation='relu'),
    BatchNormalization(),
    Maxpolining2d(2,2),
    Dropout(0.25),

    Conv2d(128,(3,3), activation='relu' ),
    BatchNormalization(),
    Maxpolining2d(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5)
    Dense(1, activation='sigmoid')

])


model.compile(
    optimizer=tf.keras.optimizers.Adan(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_wights=True)
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, factor=0.2, patience=3,min_lr=0.00001)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

model.save('cat_dog_classifier.keras')
print("Model saved as 'cat_dog_classifier.keras'")

def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_history(history)


def perdict_image(img_path, model):
    from tensorflow.keras.preprocessing import image
    import numpy as np

    img = img.load(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array,axis=0) / 255.0

    prediction = model.predict(img_array)
    cofidence = prediction[0][0]

    if confidence > 0.5:
        return f"Dog(Confidence: {confidence:.2f})"    
    else:
        return f"Dog(Confidence: {confidence:.2f})"    

print("\nTesting the model:")
try:
    test_image = [
        os.path.join(DATASET_PATH,'cat', f) for f in os.listdir(os.path.join(DATASET_PATH, 'cat'))[:3]
    ] + [
        os.path.join(DATASET_PATH,'cat', f) for f in os.listdir(os.path.join(DATASET_PATH, 'cat'))[:3]
    ]

    for img_path in test_images:
        print(f"{img_path}: {perdict_image(img_path,model)}")
except Exception as e:
    print(f"Error during testing: {e}")