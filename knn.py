from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns

def render(dataset):
  D = fetch_openml(dataset,as_frame=False)
  X = D.data[:, :2]
  y = D.target
  n_neighbors = 1
  # clf = KNeighborsClassifier()
  # clf.fit(X,y)

  # https://stackoverflow.com/questions/45075638/graph-k-nn-decision-boundaries-in-matplotlib
  # adapted  mostly from this, slight modifications
  h = 0.01

  cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
  cmap_bold = ['darkorange', 'c', 'darkblue']

  clf = KNeighborsClassifier(n_neighbors, weights='distance')
  clf.fit(X, y)

  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
  np.arange(y_min, y_max, h))
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).astype(int)
  Z = Z.reshape(xx.shape)
  plt.figure()
  plt.contourf(xx, yy, Z, cmap=cmap_light)

  sns.scatterplot(x=X[:, 0], y=X[:, 1], palette=cmap_bold, alpha=1.0, edgecolor="black", size=1)

  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

  plt.title("3-Class classification (k = %i, 'uniform' = '%s')"
  % (n_neighbors, 'uniform'))

  plt.xlabel(D.feature_names[0])
  plt.ylabel(D.feature_names[1])
  plt.show()

D = fetch_openml("one-hundred-plants-texture",as_frame=False)
X = D.data[:, :2]
y = D.target
n_neighbors = 3

scores = []
for k in range(1, 6):
  clf = KNeighborsClassifier(k, weights='uniform')
  clf.fit(X, y)
  scores.append(clf.score(X, y))

plt.plot(range(1,6), scores)
plt.show()

scores = []
for k in range(1,21,2):
  clf = KNeighborsClassifier(k, weights='distance')
  clf.fit(X, y)
  scores.append(clf.score(X, y))

plt.plot(range(1,21,2), scores)
plt.show()

render("one-hundred-plants-margin")
render("one-hundred-plants-texture")