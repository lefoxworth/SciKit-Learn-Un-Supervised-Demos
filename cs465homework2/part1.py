from sklearn.neural_network import MLPRegressor
import numpy
import pickle

x = numpy.genfromtxt("x.csv", delimiter=',')
y = numpy.genfromtxt("y.csv", delimiter=',')

data = numpy.zeros((30,30))
z = numpy.loadtxt(open("z.csv","rb"), delimiter=",")

for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = x[i] * y[j]

filename = 'learner-model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(data)
output = numpy.savetxt('z-predicted.csv', result, delimiter=',')
