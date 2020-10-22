from sklearn import mixture
import numpy
import matplotlib.pyplot as plt

data = numpy.loadtxt(open("p2-data","rb"), delimiter=",")

#plt.plot(data[:,0], data[:,1], 'bx')
#plt.axis('equal')
#plt.show()

model = mixture.GaussianMixture(n_components=5, covariance_type='diag', max_iter=500)
model.fit(data)

print("Means:\n")
print(model.means_)
print("Covariances:\n")
print(model.covariances_)

X, Y = numpy.meshgrid(numpy.linspace(-15, 15), numpy.linspace(-15,15))
XX = numpy.array([X.ravel(), Y.ravel()]).T
Z = model.score_samples(XX)
Z = Z.reshape((50,50))
 
plt.contour(X, Y, Z)
plt.scatter(data[:, 0], data[:, 1])
 
plt.show()