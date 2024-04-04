from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier


D1 = fetch_openml("one-hundred-plants-shape", as_frame=False, parser="auto")
D2 = fetch_openml("one-hundred-plants-margin", as_frame=False, parser="auto")
D3 = fetch_openml("one-hundred-plants-texture", as_frame=False, parser="auto")

clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(5), max_iter=100000)
clf.fit(D3.data, D3.target)
print(clf.score(D3.data, D3.target))
