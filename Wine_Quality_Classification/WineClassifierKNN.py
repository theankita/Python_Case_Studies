import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def Classifier(DataPath):
    Border = "-" * 40

    #Step1 : Load the dataset from CSV file

    print(Border)
    print("Step1 : Load the dataset from CSV file")
    print(Border)

    df = pd.read_csv(DataPath)

    print(Border)
    print("Some entries from the dataset : ")
    print(df.head())
    print(Border)

    #Step2 : Clean the dataset by removing empty rows

    print(Border)
    print("Step2 : Clean the dataset by removing empty rows")
    print(Border)

    df.dropna(inplace = True)
    print("Total recornds : ", df.shape[0])
    print("Total columns : ", df.shape[1])
    print(Border)

    #Step3 : Separate independent and dependent variables

    print(Border)
    print("Step3 : Separate independent and dependent variables")
    print(Border)

    X = df.drop(columns = ['Class'])
    Y = df['Class']

    print("Shape of X : ", X.shape)
    print("Shape of Y : ",Y.shape)

    print(Border)
    print("Input columns : ", X.columns.tolist)
    print("Output columns : Class")

    #Step4 : Split the data for training and testing

    print(Border)
    print("Step4 : Split the data for training and testing")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42, stratify = Y)

    print(Border)
    print("Informaation of training and testing data")
    print("X_train shape : ", X_train.shape)
    print("X_test shape : ", X_test.shape)
    print("Y_train shape : ", Y_train.shape)
    print("Y_test shape : ", Y_test.shape)
    print(Border)

    #Step5 : Feature scaling
    print(Border)
    print("Step5 : Feature scaling")
    print(Border)

    scaler = StandardScaler()

    #Independent variable scalling

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    print("Feature scaling is done")

    #Step6 : Explore the multiple values of K 
    #Hyperparameter tuning = K

    print(Border)
    print("Step6 : Explore the multiple values of K ")
    print(Border)
    
    accuracy_scores = []
    k_values = range(1,21)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train_scaled, Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_scores.append(accuracy)

    print(Border)
    print("Accuracy report of all K values from 1 to 20")

    for value in accuracy_scores:
        print(value * 100,"%")
    print(Border)

    #Step7 : Plot graph of K vs Accuracy

    print(Border)
    print("Step7 : Plot graph of K vs Accuracy")
    print(Border)

    plt.figure(figsize = (8,5))

    plt.plot(k_values, accuracy_scores, marker = 'o')

    plt.title(" K values vs Accuracy values")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(k_values))
    plt.show()

    #Step8 : Find best value of K

    print(Border)
    print("Step8 : Find best value of K")
    print(Border)

    best_k = list(k_values)[accuracy_scores.index(max(accuracy_scores))]

    print("Best of value of k is :",best_k)

    #Step9 : Build final model using best value of K

    print(Border)
    print("Step9 : Build final model using best value of K")
    print(Border)

    final_model = KNeighborsClassifier(n_neighbors = best_k)

    final_model.fit(X_train_scaled, Y_train)

    Y_pred = final_model.predict(X_test_scaled)

    #Step10 : Calculate final accuracy

    print(Border)
    print("Step10 : Calculate final accuracy")
    print(Border)

    accuracy = accuracy_score(Y_test, Y_pred)

    print("Accuracy of model is : ", accuracy*100,"%")

    #Step11 : Display confusion matrix

    print(Border)
    print("Step11 : Display confusion matrix")
    print(Border)

    cm = confusion_matrix(Y_test, Y_pred)

    print(cm)

    #Step12 : Display classification report

    print(Border)
    print("Step12 : Display classification report")
    print(Border)
    
    print(classification_report(Y_test, Y_pred))

def main():
    Border = "-" * 40

    print(Border)
    print("Wine ClassiFier Using KNN")
    print(Border)

    Classifier("WinePredictor.csv")

if __name__ == "__main__":
    main()
