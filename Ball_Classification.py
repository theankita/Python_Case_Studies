from sklearn import tree

#Rough - 1
#Smooth - 0

#Tennis -1
#Cricket - 2

def main():
    print("Ball Classification case study")

    #Original encoded data set

    #Independent Variables
    X = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0],[35, 1], [92, 0],[35, 1],[35, 1],[35, 1], [96, 0],[43, 1], [110, 0],[35, 1], [95, 0]]

    #Dependent Variables
    Y = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]

    #Independent variables for training
    Xtrain = [[35, 1], [47, 1], [90, 0], [48, 1], [90, 0],[35, 1], [92, 0],[35, 1],[35, 1],[35, 1], [96, 0],[43, 1], [110, 0]]

    #Independent variables for testing
    Xtest = [[35, 1], [95, 0]]

    #Dependent for training
    Ytrain = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2]

    #Dependent for testing
    Ytest = [1, 2]

    modelobj = tree.DecisionTreeClassifier()   #Model selection

    trainedmodel = modelobj.fit(Xtrain , Ytrain)  #This model will go on harddisk

    result = trainedmodel.predict([[35,1]]) #Output - [1 2]

    print(type(result))

    if(result == 1):
        print("Object looks like tennis ball ")

    elif(result == 2):
        print("Object looks like Cricket Ball")

if __name__ == "__main__":
    main()
