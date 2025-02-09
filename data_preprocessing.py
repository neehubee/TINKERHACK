import pandas as pd
import os
import tensorflow as tf

# Define the file paths.
csv_path = 'labels.csv'          # CSV file containing image filenames and labels.
image_dir = 'captured_images'    # Folder where your images are stored.

# Check that the CSV file exists.
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {os.path.abspath(csv_path)}")
else:
    print(f"CSV file found at: {os.path.abspath(csv_path)}")

# Load the CSV file using pandas.
df = pd.read_csv(csv_path)
print("CSV Data:")
print(df.head())

# Map each unique label (e.g., 'A', 'B') to an integer.
label_names = df['label'].unique()  # For example: array(['A', 'B'])
label_to_index = {name: index for index, name in enumerate(label_names)}
df['label_idx'] = df['label'].map(label_to_index)
print("Label mapping:", label_to_index)

# Create a list of full file paths for each image.
file_paths = [os.path.join(image_dir, fname) for fname in df['filename']]
# Create a list of corresponding numeric labels.
labels = df['label_idx'].tolist()

# Print an example to verify.
print("Example file path:", file_paths[0])
print("Example label:", labels[0])

# Create a TensorFlow dataset from the file paths and labels.
dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

# Define a function to load and preprocess images.
def load_and_preprocess_image(path, label):
    # Read the image file.
    image = tf.io.read_file(path)
    # Decode the image as PNG (use tf.image.decode_jpeg if your images are JPEGs).
    image = tf.image.decode_png(image, channels=3)
    # Resize the image to 224x224 pixels.
    image = tf.image.resize(image, [224, 224])
    # Normalize pixel values to the [0, 1] range.
    image = image / 255.0
    return image, label

# Map the preprocessing function over the dataset.
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

# Optional: Shuffle, batch, and prefetch the dataset for performance.
dataset = dataset.shuffle(buffer_size=100).batch(32).prefetch(tf.data.AUTOTUNE)

# For debugging: Print one batch's shape and labels.
for images, lbls in dataset.take(1):
    print("Images batch shape:", images.shape)
    print("Labels batch:", lbls.numpy())

if __name__ == '__main__':
    print("Data preprocessing complete .")
