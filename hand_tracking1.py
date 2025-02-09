import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

with open("hand_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                label = input("Enter gesture label (e.g., A, B, C): ")  
                writer.writerow([label] + x_coords + y_coords)

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()