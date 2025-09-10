import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import mediapipe as mp

input_root = r'C:\Users\stefa\Desktop\MSLModel\dataset'
output_root = r'C:\Users\stefa\Desktop\MSLModel\cropped_dataset'

mp_hands = mp.solutions.hands

os.makedirs(output_root, exist_ok=True)

with mp_hands.Hands(static_image_mode=True) as hands:
    for class_name in os.listdir(input_root):
        input_class_dir = os.path.join(input_root, class_name)
        output_class_dir = os.path.join(output_root, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for file in os.listdir(input_class_dir):
            img_path = os.path.join(input_class_dir, file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w, _ = image.shape
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]

                x_min = int(min(x_list) * w) - 20
                x_max = int(max(x_list) * w) + 20
                y_min = int(min(y_list) * h) - 20
                y_max = int(max(y_list) * h) + 20

                x_min, y_min = max(x_min, 0), max(y_min, 0)
                x_max, y_max = min(x_max, w), min(y_max, h)

                cropped = image[y_min:y_max, x_min:x_max]
                cv2.imwrite(os.path.join(output_class_dir, file), cropped)
