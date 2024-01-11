import os
import cv2

DATA_DIR = './dataset'
NUMBER_OF_CLASSES = 15
DATASET_SIZE = 100

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def collect_data(class_index, videocap):
    class_dir = os.path.join(DATA_DIR, str(class_index))
    create_directory(class_dir)

    print(f'Collecting data for class {class_index}')

    # Display instruction
    done = False
    while not done:
        ret, frame = videocap.read()
        cv2.putText(frame, "Если готовы, то нажмите 'А'", (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('f'):
            done = True

    counter = 0
    while counter < DATASET_SIZE:
        ret, frame = videocap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

def main():
    videocap = cv2.VideoCapture(0)

    create_directory(DATA_DIR)

    for class_index in range(NUMBER_OF_CLASSES):
        collect_data(class_index, videocap)

    videocap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




# import os
# import cv2
#
#
# DATA_DIR = './dataset'
# if not os.path.exists(DATA_DIR):
#     os.makedirs(DATA_DIR)
#
# number_of_classes = 15
# dataset_size = 100
#
#
# videocap = cv2.VideoCapture(0)
# for j in range(number_of_classes):
#     if not os.path.exists(os.path.join(DATA_DIR, str(j))):
#         os.makedirs(os.path.join(DATA_DIR, str(j)))
#
#     print('Collecting data for class {}'.format(j))
#
#     done = False
#     while True:
#         ret, frame = videocap.read()
#         cv2.putText(frame, "Если готовы, то нажмите 'А' ", (40, 100), cv2.FONT_HERSHEY_COMPLEX, 1.1, (0, 0, 0), 3,
#                     cv2.LINE_AA)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(25) == ord('f'):
#             break
#
#     counter = 0
#     while counter < dataset_size:
#         ret, frame = videocap.read()
#         cv2.imshow('frame', frame)
#         cv2.waitKey(25)
#         cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
#         counter += 1
#
# videocap.release()
# cv2.destroyAllWindows()
