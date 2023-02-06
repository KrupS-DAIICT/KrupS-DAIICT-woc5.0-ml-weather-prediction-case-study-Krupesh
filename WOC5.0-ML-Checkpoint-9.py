import numpy as np
from tkinter import *
import tkinter as tk
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier
from operator import itemgetter

# Read csv file
df = pd.read_csv('D:\SEM-04\WoC\Checkpoint 7\weather.csv')
df = pd.DataFrame(df)

input =  df[df.columns[0:4]] # Independent data
output = df['weather'] # Dependent data

# Using LinearRegression to train the model
x_train, x_test, y_train, y_test = tts(input, output, test_size=0.1)
model1 = LR(multi_class='multinomial', max_iter=1500).fit(x_train, y_train)
pred1 = model1.score(x_test, y_test)

# Model2 after using KNN 
def KN_Neighbors(X_test, X_train, Y_test, Y_train, weight):
  values = [] # List to store the K value and accuracy tuple

  # Store the accuracy score and corresponding K value in values tuple
  for i in range(5, 15):
    knn = KNeighborsClassifier(n_neighbors=i, weights=weight)
    knn.fit(X_train, Y_train)
    score = knn.score(X_test, Y_test)
    values.append([i,score])

  values = np.array(values)

  return values

model2 = KN_Neighbors(x_test, x_train, y_test, y_train, 'distance')
max_tuple = max(model2, key=itemgetter(1))
max_k = max_tuple[0]
pred2 = max_tuple[1]

# KNN using best number of neighbours
knn2 = KNeighborsClassifier(n_neighbors=int(max_k), weights='distance')
knn2.fit(x_train, y_train)

prediction=""
# Define function to predict weather
def weather_predict(precipitation_value, temp_max_value, temp_min_value, wind_value):
    ex = np.array([precipitation_value, temp_max_value, temp_min_value, wind_value])
    ex = ex.reshape(-1, 4)

    if pred1 > pred2:
        prediction = model1.predict(ex)
    else:
        prediction = knn2.predict(ex)

    Label(weather, text="Predicted weather: " + prediction, font="arial 24 bold").grid(row=8, column=2)

weather = tk.Tk()
weather.title("Weater prediction model")
weather.geometry("800x500") # Set window size

# Declare valariable type in which we take the input
Precipitation_value = tk.StringVar()
temp_max_value = tk.StringVar()
temp_min_value = tk.StringVar()
wind_value = tk.StringVar()

# Create basic structure of the frame1 to takeinputs for precipitation, temp_max, temp_min, and wind
Label(weather, text="-: Enter weather data :-", bg='cyan', font="arial 24 bold").grid(row=0, column=2)

# Labels
Precipitation = Label(weather, text="Precipitation: ", font=("arial", 16)).grid(row=1, column=1)
temp_max = Label(weather, text="temp_max: ", font=("arial", 16)).grid(row=2, column=1)
temp_min = Label(weather, text="temp_min: ", font=("arial", 16)).grid(row=3, column=1)
wind = Label(weather, text="wind: ", font=("arial", 16)).grid(row=4, column=1)

# Input boxes
Precipitation_box = Entry(weather, textvariable=Precipitation_value).grid(row=1, column=2)
temp_min_box = Entry(weather, textvariable=temp_max_value).grid(row=2, column=2)
temp_max_box = Entry(weather, textvariable=temp_min_value).grid(row=3, column=2)
wind_box = Entry(weather, textvariable=wind_value).grid(row=4, column=2)

frame1_btn = Button(weather, text="Enter", command=lambda:weather_predict(float(Precipitation_value.get()), 
                float(temp_max_value.get()), float(temp_min_value.get()), float(wind_value.get()))).grid(row=6, column=1)
frame1_btn2 = Button(weather, text="Exit", command=exit).grid(row=6, column=2)

weather.mainloop()
