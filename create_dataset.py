import os
import pickle
import cv2
import mediapipe as mp

# Initialize Mediapipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Setup Mediapipe Hands model
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.2, max_num_hands=1, min_tracking_confidence=0.4)

DATA_DIR = './dataset'

def extract_hand_landmarks(img):
    x_ = []
    y_ = []
    data_aux = []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

    return data_aux

data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    for img_path in os.listdir(label_path):
        img = cv2.imread(os.path.join(label_path, img_path))
        landmarks = extract_hand_landmarks(img)

        if landmarks:
            data.append(landmarks)
            labels.append(label)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)




# import os
# import pickle
#
# import cv2
# import mediapipe as mp
#
# # Инициализация Mediapipe для обнаружения рук
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# # Настройка модели Mediapipe Hands
# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.2, max_num_hands=1, min_tracking_confidence=0.4)
#
# DATA_DIR = './dataset'
#
# data = []
# labels = []
# for dir_ in os.listdir(DATA_DIR):
#     for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
#         data_aux = []
#
#         x_ = []
#         y_ = []
#
#         img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         results = hands.process(img_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#
#                     x_.append(x)
#                     y_.append(y)
#
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x - min(x_))
#                     data_aux.append(y - min(y_))
#
#             data.append(data_aux)
#             labels.append(dir_)
#
#
# # Сохранение данных и меток в файл pickle
# f = open('data.pickle', 'wb')
# pickle.dump({'data': data, 'labels': labels}, f)
# f.close()
