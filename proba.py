from numpy import float64, int32, int64, number
import pandas as pd

dataset = pd.read_csv('C:/RUAP-01.09/insurance.csv')
for col in dataset.columns:
    if (dataset[col].dtype == object):
        print (col, ':', dataset[col].unique())


dataset.drop(columns="charges")

from sklearn import preprocessing

def labelEncoder(dataset, col):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(dataset[col])
    return encoder.transform(dataset[col])

dataset['sex'] = labelEncoder(dataset, 'sex')
dataset['smoker'] = labelEncoder(dataset, 'smoker')
dataset['region'] = labelEncoder(dataset, 'region')

from sklearn.model_selection import train_test_split

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
dt = DecisionTreeRegressor(max_depth = 3)
rf = RandomForestRegressor(max_depth = 3, n_estimators=500)
lr.fit(x_train, y_train)
rf.fit(x_train, y_train)
dt.fit(x_train,y_train)

y_test_predicted_lr = lr.predict(x_test)
y_test_predicted_rf = rf.predict(x_test)
y_test_predicted_dt = dt.predict(x_test)

y_test_predicted_lr = lr.predict(x_test)

from sklearn.metrics import r2_score

score_lr = r2_score(y_test, y_test_predicted_lr)
score_rf = r2_score(y_test, y_test_predicted_rf)
score_dt = r2_score(y_test, y_test_predicted_dt)

print ('\n R2 Score LR : ', round(score_lr * 100), ' %')
print ('\n R2 Score RF: ', round(score_rf * 100), ' %')
print ('\n R2 Score DT: ', round(score_dt * 100), ' %')

from tkinter import *

root = Tk()
root.title("Medical insurance") # title of the GUI window
root.maxsize(900, 600) # specify the max size the window can expand to
root.config(bg="skyblue") # specify background color


label_widget = Label(root, text="Input data:", width=40)
label_widget.pack()
label_widget = Label(root, text="Age",pady=10,bg="skyblue")
label_widget.pack()
e = Entry(root, width=20)
e.pack()
label_widget = Label(root, text="Sex",pady=10,bg="skyblue")
label_widget.pack()
e_1 = Entry(root, width=20)
e_1.pack()
label_widget = Label(root, text="BMI",pady=10,bg="skyblue")
label_widget.pack()
e_2= Entry(root, width=20)
e_2.pack()
label_widget = Label(root, text="Children",pady=10,bg="skyblue")
label_widget.pack()
e_3 = Entry(root, width=20)
e_3.pack()
label_widget = Label(root, text="Smoker",pady=10,bg="skyblue")
label_widget.pack()
e_4 = Entry(root, width=20)
e_4.pack()
label_widget = Label(root, text="Region",pady=10,bg="skyblue")
label_widget.pack()
e_5 = Entry(root, width=20)
e_5.pack()

def funkcija(k):
    myLabel = Label(root, text=k, bg="skyblue")
    myLabel.pack()

def myClick():
    age = e.get()
    sex = e_1.get()
    bmi = e_2.get()
    children = e_3.get()
    smoker = e_4.get()
    region = e_5.get()
    array_ = [int64(age),int32(sex),float64(bmi),int32(children),int32(smoker),int32(region)]
    print(array_)
    k = (dt.predict([array_]))
    print(k)
    label_pred = Label(root, text="Insurance cost is:", bg="skyblue")
    label_pred.pack()
    funkcija(k)




label_1 = Label(root, text="", bg="skyblue")
label_1.pack()
myButton = Button(root, text="Predict", command= myClick, bg="white")
myButton.pack()
label_2= Label(root, text="", bg="skyblue")
label_2.pack()

root.mainloop()