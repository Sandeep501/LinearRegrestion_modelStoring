#Read the trained model from pkl file to predict the data
import pickle
import pandas
from sklearn import model_selection
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)

array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

test_size = 0.2
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

filename = '/home/sandeep/Desktop/ProjectPOC/finalized_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X_test)
predictionScore=loaded_model.score(X_test,Y_test)
print(predictionScore)
print(result[0:20])