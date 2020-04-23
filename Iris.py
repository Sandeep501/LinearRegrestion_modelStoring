import pickle
import sys
import os
import pandas
# import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

# load Iris dataset from a DataPrep package as a pandas DataFrame
iris = pandas.read_csv('irisData.csv')
print ('Iris dataset shape: {}'.format(iris.shape))

# load features and labels
X, Y = iris[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values, iris['Species'].values

# split data 65%-35% into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.35, random_state=0)

# change regularization rate and you will likely get a different accuracy
reg = 0.01

# load regularization rate from argument if present
if len(sys.argv) > 1:
   reg = float(sys.argv[1])

print("Regularization rate is {}".format(reg))

# train a logistic regression model on the training set
clf1 = LogisticRegression(C=1/reg).fit(X_train, Y_train)
print (clf1)

# evaluate the test set
accuracy = clf1.score(X_test, Y_test)
print ("Accuracy is {}".format(accuracy))

# serialize the model on disk in the special 'outputs' folder
print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(clf1, f)
f.close()