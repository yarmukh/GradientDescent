import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv('Student_Performance.csv')
#Меняем ответы 'Yes' и 'No' на '1' и '0'
encoder = LabelEncoder()
data["Extracurricular Activities"] = encoder.fit_transform(data['Extracurricular Activities'])

#Убераем колонку с "Performance Index". Получаем матрицу с значениями остальных переменных
X = data.drop(columns = "Performance Index")
#Строим столбец с значениями искомой функции
y = data['Performance Index']
#Разделяем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Подводим данные под один стандарт(старндартизуем)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Добавляем столбец свободных коэфицентов. Считаю, что должен быть не нулевой "Performance Index" при нулевых коэфицентах переменных в "X"
X_train_scaled = np.hstack([np.ones((X_train.shape[0], 1)), X_train_scaled])
X_test_scaled = np.hstack([np.ones((X_test.shape[0], 1)), X_test_scaled])

# MSE
#Целевая функция. Именно её и надо минимизировать
def compute_cost(X, y, theta):
    n = len(y)
    predictions = X.dot(theta)
    cost = (1/(n)) * np.sum(np.square(predictions - y))
    return cost

# Градиентный спуск
def gradient_descent(X, y, theta, learning_rate=0.2, iterations=1000):
    """ описание переменных в функции
        y - значения целевой переменной
        X - значения остальных переменных
        theta - коэфиценты перед переменными
        learning_rate - шаг алгоритма
        iterations - количество повторений
        вывод
        theta - искомые коэфиценты
        CostHistory - разницей между полученным значением и действительным
    """
    n = len(y)
    CostHistory = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        #вычисление градиента функции
        gradient = (1 / n) * X.T.dot(errors)
        theta -= learning_rate * gradient
        CostHistory[i] = compute_cost(X, y, theta)
        # Условие выхода из цикла
        if i > 0 and abs(CostHistory[i] - CostHistory[i - 1]) < 1e-12:
            print(f"Early stopping at iteration {i}")
            CostHistory = CostHistory[:i]
            break

    return theta, CostHistory


#Инициализация параметров
theta = np.zeros(X_train_scaled.shape[1])

theta , CostHistory = gradient_descent(X_train_scaled, y_train, theta, learning_rate = 0.1, iterations = 1000)
#Вывод
print("Оптимальные коэфиценты: ", theta[1:], '\n')
print("Свободный параметр равен: ", theta[0], '\n')
print("MSE: ",CostHistory,'\n')

#Ускоренный градиентный спуск Нестерова
def gradient_descent_with_NAG(X, y, theta, learning_rate=0.2, mu=0.9, iterations=100):
    """ описание переменных в функции
        y - значения целевой переменной
        X - значения остальных переменных
        theta - коэфиценты перед переменными
        learning_rate - шаг алгоритма
        iterations - количество повторений
        mu - коэффициент инерции
        вывод
        theta - искомые коэфиценты
        CostHistory - разницей между полученным значением и действительным
    """
    n = len(y)
    CostHistory = np.zeros(iterations)
    v = np.zeros_like(theta)

    for i in range(iterations):
        theta_forward = theta + mu * v
        predictions = X.dot(theta_forward)
        errors = predictions - y
        #Вычисление градиента функции
        gradient = (1 / n) * X.T.dot(errors)
        v = mu * v - learning_rate * gradient
        theta += v
        CostHistory[i] = compute_cost(X, y, theta)
        #Условие для выхода из цикла

        if i > 0 and abs(CostHistory[i] - CostHistory[i - 1]) < 1e-12:
            print(f"Early stopping at iteration {i}")
            CostHistory = CostHistory[:i]
            break

    return theta, CostHistory

#Инициализация параметров
theta_nag = np.zeros(X_train_scaled.shape[1])
#Запуск ускоренного градиентного спуска Нестерова
theta_nag , CostHistory_nag = gradient_descent_with_NAG(X_train_scaled, y_train, theta_nag, learning_rate = 0.1, mu = 0.47, iterations = 1000)
#Вывод
print("Оптимальные коэфиценты: ", theta_nag[1:], '\n')
print("Свободный параметр равен: ", theta_nag[0], '\n')
print("MSE ",CostHistory_nag,'\n')

#Для сравнения с функциями
model = LinearRegression()
model.fit(X_train, y_train)


#Графики

# Переобучаем LinearRegression на стандартизованных данных с intercept
model_corrected = LinearRegression(fit_intercept=False)
model_corrected.fit(X_train_scaled, y_train)
y_pred_train = model_corrected.predict(X_train_scaled)
mse_sklearn = mean_squared_error(y_train, y_pred_train)

print("MSE LinearRegression:", mse_sklearn)

# Построение графиков

#рисунок 1
plt.figure(figsize=(10, 6))
plt.plot(range(len(CostHistory)), CostHistory, color='blue', label='Gradient Descent')
plt.plot(range(len(CostHistory_nag)), CostHistory_nag, color='green', label='NAG')
plt.axhline(y=mse_sklearn, color='red', linestyle='--', label='LinearRegression MSE')
plt.xlabel('Итерации')
plt.ylabel('MSE')
plt.legend()
plt.title('рисунок 1')
plt.grid(True)
plt.show()
#рисунок 2
theta = np.zeros(X_train_scaled.shape[1])
theta , CostHistory = gradient_descent(X_train_scaled, y_train, theta, learning_rate = 0.1, iterations = 100)
theta_nag = np.zeros(X_train_scaled.shape[1])
theta_nag , CostHistory_nag = gradient_descent_with_NAG(X_train_scaled, y_train, theta_nag, learning_rate = 0.1, mu = 0.47, iterations = 100)
plt.figure(figsize=(10, 6))
plt.plot(range(len(CostHistory)), CostHistory, color='blue', label='Gradient Descent')
plt.plot(range(len(CostHistory_nag)), CostHistory_nag, color='green', label='NAG')
plt.axhline(y=mse_sklearn, color='red', linestyle='--', label='LinearRegression MSE')
plt.xlabel('Итерации')
plt.ylabel('MSE')
plt.legend()
plt.title('рисунок 2')
plt.grid(True)
plt.show()
#рисунок 3
theta = np.zeros(X_train_scaled.shape[1])
theta , CostHistory = gradient_descent(X_train_scaled, y_train, theta, learning_rate = 0.1, iterations = 10)
theta_nag = np.zeros(X_train_scaled.shape[1])
theta_nag , CostHistory_nag = gradient_descent_with_NAG(X_train_scaled, y_train, theta_nag, learning_rate = 0.1, mu = 0.47, iterations = 10)
plt.figure(figsize=(10, 6))
plt.plot(range(len(CostHistory)), CostHistory, color='blue', label='Gradient Descent')
plt.plot(range(len(CostHistory_nag)), CostHistory_nag, color='green', label='NAG')
plt.axhline(y=mse_sklearn, color='red', linestyle='--', label='LinearRegression MSE')
plt.xlabel('Итерации')
plt.ylabel('MSE')
plt.legend()
plt.title('рисунок 3')
plt.grid(True)
plt.show()
