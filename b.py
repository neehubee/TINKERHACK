import tensorflow as tf

model_path = "C:/Users/admin/Desktop/thack/my_model"  # Update the correct path
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")