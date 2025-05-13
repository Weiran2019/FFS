# -*- coding: utf-8 -*-
"""
Generating Table III, IV Figs. 7, 9a,10a
"""

from FFS_dp2 import dp2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,f1_score, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

X_train, X_test, y_train, y_test, _ = dp2("./data/train_dc23dp1.csv", "./data/test_dc23dp1.csv")

rfc = RandomForestClassifier(random_state=41)
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
f1=f1_score(y_test, y_pred, average='macro')
cr = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)
print(f"Baseline Mac F1 score: {f1}")
print("classification report: ", cr)
classes=['DE','IC','LM','NT','P','R']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
plt.figure(1)
disp.plot(
    cmap="Greens",
    colorbar=False,
    )
plt.savefig("fig7.pdf")
plt.close()

hueorder=["R","P","DE"]

pca = PCA(n_components=2)
X_embedded = pca.fit_transform(X_train)

X_embedded = X_embedded[y_train!=1]
y_train = y_train[y_train!=1]
X_embedded = X_embedded[y_train!=2]
y_train = y_train[y_train!=2]
X_embedded = X_embedded[y_train!=3]
y_train = y_train[y_train!=3]
labels=[]
for i in range(len(y_train)):
    if y_train[i] == 0:
        labels.append("DE")
    if y_train[i] == 4:
        labels.append("P")
    if y_train[i] == 5:
        labels.append("R")

plt.figure(2)
sns.set_style("white")
p=sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='Accent',hue_order=hueorder)
p.set_xlabel("PCA Component 1")
p.set_ylabel("PCA Component 2")
plt.savefig("fig9a.pdf")
plt.close()

pca = PCA(n_components=2)
X_embedded = pca.fit_transform(X_test)

X_embedded = X_embedded[y_test!=1]
y_test = y_test[y_test!=1]
X_embedded = X_embedded[y_test!=2]
y_test = y_test[y_test!=2]
X_embedded = X_embedded[y_test!=3]
y_test = y_test[y_test!=3]
labels=[]
for i in range(len(y_test)):
    if y_test[i] == 0:
        labels.append("DE")
    if y_test[i] == 4:
        labels.append("P")
    if y_test[i] == 5:
        labels.append("R")

plt.figure(3)
sns.set_style("white")
p=sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='Accent',hue_order=hueorder)
p.set_xlabel("PCA Component 1")
p.set_ylabel("PCA Component 2")
plt.savefig("fig10a.pdf")
plt.close()


