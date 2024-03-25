from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from matplotlib.pyplot import plot, show, xlabel, ylabel
from random import choices
from numpy import empty, unique

D = fetch_openml("one-hundred-plants-shape", as_frame=False)
M = SGDClassifier()
scores = empty((2500,))
for i in range(len(scores)):
    indices = choices(range(D.data.shape[0]), k=500)
    M.partial_fit(D.data[indices,:], D.target[indices], classes=unique(D.target))
    scores[i] = M.score(D.data, D.target)
print(scores)
plot(range(len(scores)), scores, label='batch size=50')
xlabel('Number of iterations', fontsize=16)
ylabel('Accuracy', fontsize=16)
show()
