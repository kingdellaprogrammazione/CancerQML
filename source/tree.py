import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from pathlib import Path


pure_file_path = Path("./data/diabetesrenewed.csv")


pima = pd.read_csv(pure_file_path)


categories = pima.columns.tolist()
labels = categories[:-1]
X = pima[labels]
outcomes = categories[-1:]
y = pima[outcomes]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 1)
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


y_pred = clf.predict(X_test)
yaccuracy = metrics.accuracy_score(y_test, y_pred)
print(yaccuracy)


print(labels)
print(outcomes)


plt.figure(figsize = (12, 8))
plot_tree(clf,feature_names = labels, filled = True, max_depth = 3, fontsize = 8)
plt.show()
