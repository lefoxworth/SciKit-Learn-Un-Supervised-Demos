from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy
import pickle

x = numpy.genfromtxt("x.csv", delimiter=',')
y = numpy.genfromtxt("y.csv", delimiter=',')

data = numpy.zeros((30,30))
z = numpy.loadtxt(open("z.csv","rb"), delimiter=",")

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = x[i] * y[j]

X_train, X_test, y_train, y_test = train_test_split(data, z, random_state=1)
model = MLPRegressor(hidden_layer_sizes=(3,3,3), solver="lbfgs", random_state=1, max_iter=500).fit(X_train, y_train)
filename = 'learner-model.sav'
pickle.dump(model, open(filename,'wb'))
model.predict(X_test)
print(model.score(X_test, y_test))