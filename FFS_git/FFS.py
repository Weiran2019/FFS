# -*- coding: utf-8 -*-
"""
Generating Table II, III, IV Figs. 2,3,4,5,8,9b,10b
"""
from FFS_dp2 import dp2
from sklearn.feature_selection import chi2,SelectKBest,f_classif
from sklearn.metrics import classification_report,f1_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import time

def myplot(x,y,num,xs,ys,path="fig.pdf",issubfigure=True):
    fig = plt.figure(num=num)

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8

    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.set_xlabel('k',fontsize=10)
    ax1.set_ylabel('flowspectrum',fontsize=10)
    ax1.plot(x, y, marker='.',linestyle='--')
    ax1.scatter(xs,ys,marker='*',color='r')
    ax1.ticklabel_format(style='sci',scilimits=(0,2),axis='y',useMathText=True)
    
    if issubfigure==True:
   
        left, bottom, width, height = 0.5, 0.5, 0.3, 0.3

        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.set_xlabel('k',fontsize=10)
        ax2.set_ylabel('flowspectrum',fontsize=10)
        ax2.plot(x[50:], y[50:], marker='.',linestyle='--')
        ax2.scatter(xs,ys,marker='*',color='r')
        ax2.ticklabel_format(style='sci',scilimits=(0,2),axis='y',useMathText=True)
    plt.savefig(path)

def myfeatures(indices_all,k):
    indices=[]
    for i in range(k):
        if i==0:
            indices.append(indices_all[i][i])
        else:
            for element in set(indices_all[i])-set(indices_all[i-1]):
                indices.append(element)
    return indices
    

def myfeatsele(X_train,y_train,score_f=chi2):
    score1=[]
    score2=[]
    indices_all = []
    for i in range(len(X_train[0])):
        new_data = SelectKBest(score_f, k=i+1).fit(X_train, y_train)
        indices = new_data.get_support(indices=True)
        x_train = new_data.transform(X_train)
        nfs1 = davies_bouldin_score(x_train, y_train)
        nfs2 = calinski_harabasz_score(x_train, y_train)
        score1.append(nfs1)
        score2.append(nfs2)
        indices_all.append(indices)
        print(f"{i+1}/{len(X_train[0])}")
    return score1, score2, indices_all

X_train,X_test,y_train,y_test,columns_names = dp2("./data/train_dc23dp1.csv","./data/test_dc23dp1.csv")
print("SCVIC-APT-2021 Datasets Feature names After DCDP: ", columns_names)

score1, score2, indices_allf = myfeatsele(X_train, y_train, score_f=f_classif)  
myplot(range(1,len(score1)+1),score1,1,61,score1[60],path="fig3.pdf")
myplot(range(1,len(score2)+1),score2,2,57,score2[56],path="fig5.pdf")

start=time.time()
score1, score2, indices_allc = myfeatsele(X_train, y_train, score_f=chi2)
end=time.time()
tottime=end-start
print(f"Running time of FFS:{tottime}s")
myplot(range(1,len(score1)+1),score1,3,56,score1[55],path="fig2.pdf")
myplot(range(1,len(score2)+1),score2,4,60,score2[59],path="fig4.pdf")

k=60
indices = indices_allf[k]
print("ANOVA DBI Selected Features",indices)
new_train = X_train[:,indices]
new_test = X_test[:,indices]
rfc = RandomForestClassifier(random_state=41)
rfc.fit(new_train,y_train)

y_pred = rfc.predict(new_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"ANOVA DBI Mac F1: {f1:2f}")

k=56
indices = indices_allf[k]
print("ANOVA CHI Selected Features",indices)
new_train = X_train[:,indices]
new_test = X_test[:,indices]
rfc = RandomForestClassifier(random_state=41)
rfc.fit(new_train,y_train)

y_pred = rfc.predict(new_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"ANOVA CHI Mac F1: {f1:2f}")

k=59
indices = indices_allc[k]
print("chi2 CHI Selected Features",indices)
new_train = X_train[:,indices]
new_test = X_test[:,indices]
rfc = RandomForestClassifier(random_state=41)
rfc.fit(new_train,y_train)

y_pred = rfc.predict(new_test)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"chi2 CHI Mac F1: {f1:2f}")

k=55
indices = indices_allc[k]
print("chi2 DBI Selected Features",indices)
new_train = X_train[:,indices]
new_test = X_test[:,indices]
rfc = RandomForestClassifier(random_state=41)
rfc.fit(new_train,y_train)

y_pred = rfc.predict(new_test)
f1 = f1_score(y_test, y_pred, average='macro')
cr = classification_report(y_test, y_pred, digits=4)
cm = confusion_matrix(y_test, y_pred)
print(f"chi2 DBI Mac F1: {f1:2f}")
print("chi2 DBI Classification Report: ", cr)
classes=['DE','IC','LM','NT','P','R']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
plt.figure(5)
disp.plot(
    cmap="Greens",
    colorbar=False,
    )
plt.savefig("fig8.pdf")
plt.close()

pca = PCA(n_components=2)
X_embedded = pca.fit_transform(new_train)

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

hueorder=["R","P","DE"]

plt.figure(6)
sns.set_style("white")
p=sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='Accent',hue_order=hueorder)
p.set_xlabel("PCA Component 1")
p.set_ylabel("PCA Component 2")
plt.savefig("fig9b.pdf")
plt.close()

pca = PCA(n_components=2)
X_embedded = pca.fit_transform(new_test)

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

hueorder=["R","P","DE"]

plt.figure(7)
sns.set_style("white")
p=sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='Accent',hue_order=hueorder)
p.set_xlabel("PCA Component 1")
p.set_ylabel("PCA Component 2")
plt.savefig("fig10b.pdf")
plt.close()
