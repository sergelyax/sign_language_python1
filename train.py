import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def load_data(file_path):
    with open(file_path, 'rb') as file:
        data_dictionary = pickle.load(file)
    return data_dictionary['data'], data_dictionary['labels']

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_predict, y_test)
    return accuracy

def save_model(model, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump({'trained_model': model}, file)

def main():
    data_path = './data.pickle'
    model_path = 'trained_model.p'

    # Загрузка данных
    data, labels = load_data(data_path)

    # Разделение данных на обучающую и тестовую выборки
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Обучение модели
    model = train_random_forest(x_train, y_train)

    # Оценка точности модели
    accuracy = evaluate_model(model, x_test, y_test)
    print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

    # Сохранение обученной модели
    save_model(model, model_path)

if __name__ == "__main__":
    main()






# import pickle
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
#
#
# data_dictionary = pickle.load(open('./data.pickle', 'rb'))
#
# data = np.asarray(data_dictionary['data'])
# labels = np.asarray(data_dictionary['labels'])
#
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#
# model = RandomForestClassifier()
# model.fit(x_train, y_train)
# y_predict = model.predict(x_test)
# # оценка точности модели
# score = accuracy_score(y_predict, y_test)
#
# print('{}% of samples were classified correctly !'.format(score * 100))
#
# f = open('trained_model.p', 'wb')
# pickle.dump({'trained_model': model}, f)
# f.close()
