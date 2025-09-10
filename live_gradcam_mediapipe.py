import cv2
import numpy as np
import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from get_model import get_model
from PIL import Image
import mediapipe as mp

MODEL_NAME = 'mobilenet'
# MODEL_PATH = f'final_model_{MODEL_NAME}.pth'
MODEL_PATH = f'best_model_{MODEL_NAME}.pth'
CLASS_NAMES = [
    'A', 'B', 'C', 'Ch', 'D', 'Dz', 'Dzh', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'Sh', 'T', 'U', 'V', 'Z', 'Zh'
]

model = get_model(MODEL_NAME, len(CLASS_NAMES)).to("cpu")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

if MODEL_NAME == 'mobilenet':
    target_layers = [model.features[-1]]
elif MODEL_NAME == 'resnet':
    target_layers = [model.layer4[-1]]
elif MODEL_NAME == 'efficientnet':
    target_layers = [model.features[-1]]


cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    crop_img = rgb_frame
    h, w, _ = rgb_frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            x_min = int(min(x_coords) * w) - 20
            x_max = int(max(x_coords) * w) + 20
            y_min = int(min(y_coords) * h) - 20
            y_max = int(max(y_coords) * h) + 20

            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            crop_img = rgb_frame[y_min:y_max, x_min:x_max]
            break

    pil_img = Image.fromarray(crop_img)
    resized_img = pil_img.resize((224, 224))
    input_tensor = transform(resized_img).unsqueeze(0).to("cpu")

    original_img = np.array(resized_img) / 255.0

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(prob, 1)
        label = CLASS_NAMES[pred_idx.item()]
        conf = confidence.item()

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_image = show_cam_on_image(original_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    text = f"Predicted: {label} ({conf * 100:.1f}%)"
    cv2.putText(cam_image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Grad-CAM Live", cam_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()
