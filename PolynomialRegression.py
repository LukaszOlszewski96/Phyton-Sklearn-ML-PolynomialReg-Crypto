import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, r2_score


#Pobranie danych z pliku csv
bitcoin = pd.read_csv('C:/Users/lukas/Desktop/Programowanie/Magisterka/BTC-USD-Month.csv')
bitcoin = bitcoin.dropna() #wartości null zostaną usunięte

#Sprawdzenie zestawu danych, czy zawiera kilka nieznanych wartości oraz zostaną one usunięte przez "dropna()"
bitcoin.isnull().sum()
bitcoin.dropna(inplace=True)


#Wybranie danych jakie będą urzyte do uczenia, a które do testowania:
#30% - dane do testów
#70% - dane do uczenia
X = ["Open", "High", "Low", "Volume"]
Y = "Close"

x_train, x_test, y_train, y_test = train_test_split(
bitcoin[X],
bitcoin[Y],
test_size = .15,
random_state=0
)

#Predict prices:
#Predykcja na 7 dni
future_set = bitcoin.shift(periods = 0).tail(7)
print(future_set)

#Regresja wielomianowa:
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(x_train)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)

#Predict prices:
prediction_poly = pol_reg.predict(poly_reg.fit_transform(future_set[X]))

#Wydruk danych porównawczych w tabeli(Realne i Przewidziane):
df2 = pd.DataFrame({'Real Values' :future_set['Close'], 'Predicted Values' :prediction_poly})
print(df2)
#Plotowanie na wykresie
plt.plot(bitcoin["Date"], bitcoin["Close"], color='goldenrod', lw=2)
plt.plot(future_set["Date"], prediction_poly , color='b')
plt.show()

print('Score dla wielo: ', r2_score(future_set['Close'],prediction_poly))
