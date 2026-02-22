import pandas as pd 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

Border = "-"*63

#########################################################################
#   Step1 : Load the dataset
#########################################################################

print(Border)
print("Step1 : Load the dataset")
print(Border)

DatasetPath = "iris.csv"

df = pd.read_csv(DatasetPath)

print("Dataset gets loaded successfully...")

print("Initial entries from dataset : ")

print(df.head())

#########################################################################
#   Step2 : Data Analysis (EDA)
#########################################################################

print(Border)
print("Step2 : Data analysis")
print(Border)

print("Shape of dataset : ", df.shape)

print("Column name : ", list(df.columns))

print("Missing values (Per Column)")

print(df.isnull().sum())

print("Class distribution (Species count)")
print(df["species"].value_counts())

print("Statistical report of dataset")
print(df.describe())

#########################################################################
#   Step3 : Decide Independent and Dependent Variables
#########################################################################

print(Border)
print("Step3 : Data analysis")
print(Border)

#X : Independent Variables - Features
#Y : Dependent Varibles - Lables

feature_cols = [
    "sepal.length(cm)",
    "sepal.width(cm)",
    "petal.length(cm)",
    "petal.width(cm)"
]

X = df[feature_cols]
Y = df["species"]

print("X shape : ", X.shape)
print("Y shape : ", Y.shape)

#########################################################################
#   Step4 : Visualisation of dataset
#########################################################################

print(Border)
print("Step4 : Visualisation of dataset")
print(Border)

#Scatter plot

plt.figure(figsize = (7,5))

for sp in df["species"].unique():
    temp = df[df["species"] == sp]
    plt.scatter(temp["petal.length(cm)"],temp["petal.width(cm)"], label = sp)

plt.title("Iris : Petel length vs Petal width")
plt.xlabel("")

plt.legend() #To show species in a box
plt.grid(True)
plt.show()

#########################################################################
#   Step5 : Split the dataset training and testing
#########################################################################

print(Border)
print("Step5 : Split the dataset training and testing")
print(Border)

#Test size = 20%
#Train size = 80%

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size = 0.5,
    random_state = 42   #It will give data as previous data   
)

print("Spliting activity done : ")

print("X - Independent : ", X.shape) #150, 4
print("Y - Dependent : ", Y.shape) #150

print("X_train : ",X_train.shape) #120 , 4
print("X_test : ",X_test.shape) #30 , 4 

print("Y_train : ",Y_train.shape) #120
print("Y_test : ",Y_test.shape) #30

#########################################################################
#   Step6 : Build the model
#########################################################################

print(Border)
print("Step6 : Build the model")
print(Border)

print("We are going to use DecisionTreeClassifier")

model = DecisionTreeClassifier(
    criterion = "gini",
    max_depth = 5,
    random_state = 42
)

print("Model succsesfully created : ", model)

#########################################################################
#   Step7 : Train the model
#########################################################################

print(Border)
print("Step7 : Train the model")
print(Border)

model.fit(X_train, Y_train)

print("Model training completed")

#########################################################################
#   Step8 : Evaluate the model
#########################################################################

print(Border)
print("Step8 : Evaluate the model")
print(Border)

Y_pred = model.predict(X_test)

print("Model evaluation (testing) complete")

print(Y_pred.shape)

print("Expected answers : ")
print(Y_test)

print("Predicated answers : ")
print(Y_pred)

#########################################################################
#   Step9 : Evaluate the model performance
#########################################################################

print(Border)
print("Step9 : Evaluate the model performance")
print(Border)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy of a model : ", accuracy*100 , "%")

cm = confusion_matrix(Y_test, Y_pred)
print("Confusion matrix : ")
print(cm)

print("Classification Report")

print(classification_report(Y_test, Y_pred))

#########################################################################
#   Step10 : Plot confusion matrxi
#########################################################################

print(Border)
print("Step10 : Plot confusion matrxi")
print(Border)

data = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = model.classes_)
data.plot()

plt.title("Confusion matrix of iris dataset")
plt.show()
