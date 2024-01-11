import sys
import pickle
import cv2
import mediapipe as mp
import numpy as np
import schedule
from PyQt5.QtCore import Qt, QTimer, QTranslator
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QDialog, QFormLayout, \
    QComboBox, QDialogButtonBox


class LanguageDialog(QDialog):
    def __init__(self, parent=None):
        super(LanguageDialog, self).__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr('Choose Language'))
        layout = QVBoxLayout()

        self.language_label = QLabel(self.tr('Choose Language:'))
        self.language_combo = QComboBox()
        self.language_combo.addItem(self.tr('English'))
        self.language_combo.addItem(self.tr('Russian'))
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.setLayout(layout)

    def get_selected_language(self):
        return self.language_combo.currentText()


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr('Settings'))
        layout = QFormLayout()

        self.camera_label = QLabel(self.tr('Select Camera:'))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem('0')  # Assuming 0 is the default camera
        # Add more cameras if needed
        layout.addRow(self.camera_label, self.camera_combo)

        self.model_label = QLabel(self.tr('Select Model:'))
        self.model_combo = QComboBox()
        # Add available models
        self.model_combo.addItem('Model 1')
        self.model_combo.addItem('Model 2')
        layout.addRow(self.model_label, self.model_combo)

        self.save_button = QPushButton(self.tr('Save'))
        self.save_button.clicked.connect(self.save_settings)
        layout.addRow(self.save_button)

        self.setLayout(layout)

    def save_settings(self):
        selected_camera = self.camera_combo.currentText()
        selected_model = self.model_combo.currentText()

        print(f"Selected Camera: {selected_camera}")
        print(f"Selected Model: {selected_model}")

        self.accept()


class GestureRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.word = ''
        self.temp = ''
        self.add_letter = True
        self.letter_dict = {0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ж', 7: 'З', 8: 'И', 9: 'Й', 10: 'К', 11: 'Л',
                            12: 'М', 13: 'Н', 14: 'О'}
        self.predicted_character = ''
        self.recognition_started = False

        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)

        self.word_label = QLabel(self.tr('Recognized Word:'))
        self.word_label.setAlignment(Qt.AlignCenter)
        self.word_label.setStyleSheet("font-size: 20pt; font-weight: bold;")

        self.start_button = QPushButton(self.tr('Start'))
        self.start_button.clicked.connect(self.start_recognition)

        self.stop_button = QPushButton(self.tr('Stop'))
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)

        self.clear_button = QPushButton(self.tr('Clear'))
        self.clear_button.clicked.connect(self.clear_word)

        self.settings_button = QPushButton(self.tr('Settings'))
        self.settings_button.clicked.connect(self.show_settings_dialog)

        self.language_button = QPushButton(self.tr('Change Language'))
        self.language_button.clicked.connect(self.change_language)

        self.close_button = QPushButton(self.tr('Close'))
        self.close_button.clicked.connect(self.close_application)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.video_label)
        layout.addWidget(self.word_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.settings_button)
        layout.addWidget(self.language_button)
        layout.addWidget(self.close_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.translator = QTranslator()

        self.app = QApplication.instance()

        self.videocap = cv2.VideoCapture(0)
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1,
                                              min_tracking_confidence=0.5)

        model_path = './trained_model.p'  # Replace with your actual model path
        self.model = self.load_model(model_path)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle(self.tr('Gesture Recognition App'))

    def load_model(self, model_path):
        model_dict = pickle.load(open(model_path, 'rb'))
        return model_dict['trained_model']

    def draw_hand_landmarks(self, hand_landmarks, letter, x_min, y_min, x_max, y_max):
        mp.solutions.drawing_utils.draw_landmarks(
            self.frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style())

        cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 4)
        cv2.putText(self.frame, letter, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    def update_frame(self):
        data_aux = []
        x_ = []
        y_ = []

        ret, self.frame = self.videocap.read()

        height, weight, _ = self.frame.shape

        # Detection of hands in the frame
        results = self.hands.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

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

            prediction = self.model.predict([np.asarray(data_aux)])
            self.predicted_character = self.letter_dict[int(prediction[0])]

            schedule.run_pending()

            self.draw_hand_landmarks(results.multi_hand_landmarks[0], self.predicted_character, x_min, y_min, x_max, y_max)

            if self.add_letter:
                schedule.every(3).seconds.do(self.check_word)
                self.add_letter = False

        h, w, c = self.frame.shape
        q_image = QImage(self.frame.data, w, h, w * c, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def check_word(self):
        self.word += self.predicted_character
        self.temp = ''  # сбросим временную переменную
        self.word_label.setText(self.tr("Recognized Word: ") + self.word)

    def clear_word(self):
        self.word = ''
        self.add_letter = True  # сбросим флаг
        self.word_label.setText(self.tr("Recognized Word: "))

    def close_application(self):
        self.stop_recognition()
        self.videocap.release()
        self.close()

    def start_recognition(self):
        if not self.recognition_started:
            self.recognition_started = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.clear_button.setEnabled(True)
            self.settings_button.setEnabled(False)
            self.language_button.setEnabled(False)
            self.timer.start(30)

    def stop_recognition(self):
        if self.recognition_started:
            self.recognition_started = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self.settings_button.setEnabled(True)
            self.language_button.setEnabled(True)
            self.timer.stop()

    def show_settings_dialog(self):
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec_()

    def change_language(self):
        dialog = LanguageDialog(self)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            selected_language = dialog.get_selected_language()

            if selected_language == 'English':
                self.translate_english()
            elif selected_language == 'Russian':
                self.translate_russia()
            else:
                return

    def translate_russia(self):
        self.setWindowTitle(self.tr('Приложение для распознования жестов'))
        self.start_button.setText(self.tr('Старт'))
        self.stop_button.setText(self.tr('Стоп'))
        self.clear_button.setText(self.tr('Очистить'))
        self.settings_button.setText(self.tr('Настройки'))
        self.language_button.setText(self.tr('Сменить язык'))
        self.close_button.setText(self.tr('Закрыть'))
        self.word_label.setText(self.tr('Распознанное слов: ') + self.word)

    def translate_english(self):
        self.setWindowTitle(self.tr('Gesture Recognition App'))
        self.start_button.setText(self.tr('Start'))
        self.stop_button.setText(self.tr('Stop'))
        self.clear_button.setText(self.tr('Clear'))
        self.settings_button.setText(self.tr('Settings'))
        self.language_button.setText(self.tr('Change Language'))
        self.close_button.setText(self.tr('Close'))
        self.word_label.setText(self.tr('Recognized Word: ') + self.word)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = GestureRecognitionApp()
    main_window.show()
    sys.exit(app.exec_())
