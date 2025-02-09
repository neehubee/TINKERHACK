import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model_path = r"C:\Users\admin\Desktop\thack\model"  # Update with correct path
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# Define test image path
img_path = r"C:\Users\admin\Desktop\thack\test3.jpg"  # Update this

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))  # Ensure this matches training size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Normalize only if model was trained with normalization
img_array = img_array / 255.0  # Uncomment if needed

# Make prediction
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)  # Get class index
confidence_score = np.max(predictions)  # Get highest probability

# Confidence threshold (Optional)
confidence_threshold = 0.5
if confidence_score >= confidence_threshold:
    print(f"✅ Predicted Class: {predicted_class_index} (Confidence: {confidence_score:.2f})")
else:
    print("⚠ Low confidence prediction. Try a different image.")
    predicted_class_index = np.argmax(predictions)
confidence_score = np.max(predictions)
print(f"Predicted Class: {predicted_class_index}, Confidence: {confidence_score:.2f}")