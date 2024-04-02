from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.pyplot import plot, show, xlabel, ylabel
from random import choices
from numpy import empty, unique

D1 = fetch_openml("one-hundred-plants-shape", as_frame=False)
D2 = fetch_openml("one-hundred-plants-margin", as_frame=False)
D3 = fetch_openml("one-hundred-plants-texture", as_frame=False)

M = SGDClassifier()
M.fit(D1.data, D1.target)
print(f"Score on shape: {M.score(D1.data, D1.target)}")

M = SGDClassifier()
M.fit(D2.data, D2.target)
print(f"Score on margin: {M.score(D2.data, D2.target)}")

M = SGDClassifier()
M.fit(D3.data, D3.target)
print(f"Score on texture: {M.score(D3.data, D3.target)}")

degree = 3
print("Calculating polynomial features of D1_poly")
D1_poly = PolynomialFeatures(degree).fit_transform(D1.data) 
print("Calculating polynomial features of D2_poly")
D2_poly = PolynomialFeatures(degree).fit_transform(D2.data) 
print("Calculating polynomial features of D3_poly")
D3_poly = PolynomialFeatures(degree).fit_transform(D3.data) 

print("Starting classifiers on polynomial features")
M = SGDClassifier()
print("fitting on D1_poly")
M.fit(D1_poly.data, D1.target)
print(f"Score on shape (poly): {M.score(D1_poly.data, D1.target)}")

M = SGDClassifier()
print("fitting on D2_poly")
M.fit(D2_poly.data, D2.target)
print(f"Score on margin (poly): {M.score(D2_poly.data, D2.target)}")

M = SGDClassifier()
print("fitting on D3_poly")
M.fit(D3_poly.data, D3.target)
print(f"Score on texture (poly): {M.score(D3_poly.data, D3.target)}")
