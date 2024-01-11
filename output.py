import pickle
import schedule
import cv2
import mediapipe as mp
import numpy as np


def load_model(model_path):
    model_dict = pickle.load(open(model_path, 'rb'))
    return model_dict['trained_model']


def draw_hand_landmarks(frame, hand_landmarks, letter, word, x_min, y_min, x_max, y_max):
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 4)
    cv2.putText(frame, letter, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, word, (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3,
                         cv2.LINE_AA)

word = ''
temp = ''
def main():
    global mp_hands, mp_drawing, mp_drawing_styles
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    model_path = './trained_model.p'
    model = load_model(model_path)

    videocap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1,
                           min_tracking_confidence=0.5)

    letter_dict = {0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ж', 7: 'З', 8: 'И', 9: 'Й', 10: 'К', 11: 'Л',
                   12: 'М', 13: 'Н', 14: 'О'}

    add_letter = True


    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = videocap.read()

        height, weight, _ = frame.shape

        # Detection of hands in the frame
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x_min = int(min(x_) * weight) - 10
            y_min = int(min(y_) * height) - 10

            x_max = int(max(x_) * weight) - 10
            y_max = int(max(y_) * height) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = letter_dict[int(prediction[0])]


            schedule.run_pending()

            def check_word(temp):
                global word
                temp += predicted_character
                word += temp

            draw_hand_landmarks(frame, results.multi_hand_landmarks[0], predicted_character, word, x_min, y_min, x_max, y_max)

            if add_letter:
                schedule.every(3).seconds.do(check_word, temp)
                add_letter = False



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('frame', frame)

    videocap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# import pickle
#
# import schedule
# import cv2
# import mediapipe as mp
# import numpy as np
#
# model_dict = pickle.load(open('./trained_model.p.', 'rb'))
# model = model_dict['trained_model']
# videocap = cv2.VideoCapture(0)
#
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
#
# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1, min_tracking_confidence=0.5)
#
# letter_dict = {0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ж', 7: 'З', 8: 'И', 9: 'Й', 10: 'К', 11: 'Л', 12: 'М', 13: 'Н', 14: 'О'}
#
# add_letter = True
# word = ''
# temp = ''
# while True:
#     data_aux = []
#     x_ = []
#     y_ = []
#
#     ret, frame = videocap.read()
#
#     height, weight, _ = frame.shape
#
#     # detection of hands in the frame
#     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#
#             mp_drawing.draw_landmarks(
#                 frame,
#                 # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#
#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#
#                 x_.append(x)
#                 y_.append(y)
#
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))
#
#         x_min = int(min(x_) * weight) - 10
#         y_min = int(min(y_) * height) - 10
#
#         x_max = int(max(x_) * weight) - 10
#         y_max = int(max(y_) * height) - 10
#
#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_character = letter_dict[int(prediction[0])]
#
#         def check_word(temp):
#             global word
#             temp += predicted_character
#             word += temp
#
#
#         cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 4)
#         cv2.putText(frame, predicted_character, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)
#         cv2.putText(frame, word, (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3,
#                     cv2.LINE_AA)
#
#         schedule.run_pending()
#
#         if add_letter:
#             schedule.every(3).seconds.do(check_word, temp = temp)
#             add_letter = False
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     cv2.imshow('frame', frame)
#
# videocap.release()
# cv2.destroyAllWindows()
#
