import cv2
import os

# Create an output folder if it doesn't exist
output_folder = "captured_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the webcam (change the index if necessary)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

img_counter = 0
print("Press 'c' to capture an image, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the live video feed
    cv2.imshow("Data Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save the current frame as an image file
        img_name = os.path.join(output_folder, f"gesture_{img_counter}.png")
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")
        img_counter += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()