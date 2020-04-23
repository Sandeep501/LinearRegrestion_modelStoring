# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)


X, Y = dataframe[['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']].values, dataframe['class'].values

test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit the model on training set
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)

# predictDiabetClass=model.predict(X_test)
# # pridict the diabeties class for 10 records in the dataset
# print(model.score(X_test,Y_test))
# print(predictDiabetClass[:10])


# save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(model, open(filename, 'wb'))

