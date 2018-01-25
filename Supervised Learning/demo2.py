import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import learning_curve

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import tree

train=pd.read_csv('input/GGG.csv')
train = train.drop(['id','color'], axis=1)

#print train.head()
#print "Dataset Length: ", len(train)
#print "Dataset Shape: ", train.shape


#splitting features and target from dataset
X = train.values[:, 0:3]
Y = train.values[:,4]

#manual split of 30% test and 70% training data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#PLOTTING TO SEE WHICH DEPTH IS BEST
d_range = range(1, 15)
scores1 = []
errors1 = []
for d in d_range:
    dtc = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=d, min_samples_leaf=1)
    dtc.fit(X_train, y_train)
    y_dpred = dtc.predict(X_test)
    scores1.append(accuracy_score(y_test,y_dpred))
    errors1.append(mean_squared_error(y_test,y_dpred))

#plt.plot(d_range, scores1)
#plt.plot(d_range, errors1)
#plt.title("Decision Tree accuracy vs. various depths")
#plt.xlabel('Max depth of tree = blue, MSE = yellow')
#plt.ylabel('Testing accuracy score')
#plt.show()

#Decision Tree Classifier with criterion gini index and max_depth 3 (best)
clf_gini1 = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=4, min_samples_leaf=1)
clf_gini1.fit(X_train, y_train)
#print "Accuracy for Decision Tree on training set", clf_gini1.score(X_train, y_train)

#prediction
y_pred1 = clf_gini1.predict(X_test)
y1 = clf_gini1.predict(X_train)

#print y_pred1
print "Training accuracy for Decision Tree with max_depth 4 is: ", accuracy_score(y_train,y1)*100
print "Testing accuracy for Decision Tree with max_depth 4 is: ", accuracy_score(y_test,y_pred1)*100
print 'Training Mean Squared error for DT ', mean_squared_error(y_train, y1)
print 'Testing Mean Squared error for DT ', mean_squared_error(y_test, y_pred1)


def prune(decisiontree, min_samples_leaf = 1):
    if decisiontree.min_samples_leaf >= min_samples_leaf:
        raise Exception('Tree already more pruned')
    else:
        decisiontree.min_samples_leaf = min_samples_leaf
        tree = decisiontree.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                tree.children_left[i]=-1
                tree.children_right[i]=-1

#Pruning clf1
prune(clf_gini1, min_samples_leaf = 7)
y_pred3 = clf_gini1.predict(X_test)
#print y_pred3
print "Accuracy for Decision Tree with max_depth 3 with pruning leaf produces same accurcy of: ", accuracy_score(y_test,y_pred3)*100



#ADABOOSTING THE DECISION TREE==================================================

from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier(
    clf_gini1,
    n_estimators=2,
    algorithm="SAMME")
boost.fit(X_train, y_train)
b1_pred = boost.predict(X_train)
b_pred = boost.predict(X_test)


#varied estimator
est_range = range(1,30)
scoresada = []
for e in est_range:
    b = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=e, learning_rate=1.5, algorithm="SAMME")
    b.fit(X_train, y_train)
    bpred = b.predict(X_test)
    scoresada.append(accuracy_score(y_test,bpred))
#plt.title("AdaBoosted Decision Tree accuracy vs. varying estimators")
#plt.plot(est_range, scoresada)
#plt.xlabel('Value of estimator')
#plt.ylabel('Testing accuracy score')
#plt.show()

print 'Accuracy for boosted DT on training set', accuracy_score(y_train, b1_pred)*100
print 'Accuracy for boosted DT on testing set', accuracy_score(y_test, b_pred)*100
print 'Mean Squared error for boosted DT training ', mean_squared_error(y_train, b1_pred)
print 'Mean Squared error for boosted DT testing', mean_squared_error(y_test, b_pred)


#K Nearest Neighbor ============================================
from sklearn.neighbors import KNeighborsClassifier

#PLOTTING BASED ON WHICH N = NEIGHBOR IS BEST
k_range = range(1, 25)
scores = []
err = []
for k in k_range:
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X_train, y_train)
    y_kpred = kn.predict(X_test)
    scores.append(accuracy_score(y_test,y_kpred))
    err.append(mean_squared_error(y_test,y_kpred))
#plt.title("KNN accuracy vs. number of neighbors, GGG Dataset")
#plt.plot(k_range, scores)
#plt.plot(k_range, err)
#plt.xlabel('Value of k for KNN')
#plt.ylabel('Testing accuracy score')
#plt.show()

from sklearn.metrics import confusion_matrix
    
#print 'It looks like highest accuracy was achieved with n=19'
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_train, y_train)
y_p = knn.predict(X_train)

y_pred5 = knn.predict(X_test)
#print y_pred5
print "Accuracy for KNN with n=19: ", accuracy_score(y_test,y_pred5)*100

print 'Training Mean Squared error for KNN n=19 ', mean_squared_error(y_train, y_p)
print 'Testing Mean Squared error for KNN n=19', mean_squared_error(y_test, y_pred5)

x = pd.DataFrame(
    confusion_matrix(y_test, y_pred5),
    columns=['Predicted Ghost', 'Predicted Ghoul', 'Predicted Goblin'],
    index=['True Ghost', 'True Ghoul','True Goblin']
)

print x

#SVC ============================================
from sklearn.svm import SVC

#LINEAR KERNEL
C_range = range(1, 20)
scores3 = []
for c in C_range:
    lin = SVC(kernel='linear', C=c)
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    scores3.append(accuracy_score(y_test,lin_pred))

#plt.plot(C_range, scores3)
#plt.xlabel('Value of C')
#plt.ylabel('Testing accuracy score')
#plt.show()

#POLY KERNEL
deg_range = range(1, 20)
scores2 = []
for deg in deg_range:
    poly = SVC(kernel='poly', degree=deg)
    poly.fit(X_train, y_train)
    poly_pred = poly.predict(X_test)
    scores2.append(accuracy_score(y_test,poly_pred))

#plt.plot(deg_range, scores2)
#plt.xlabel('Value of degree')
#plt.ylabel('Testing accuracy score')
#plt.show()


l = SVC(kernel='linear', C=2)
l.fit(X_train, y_train)
lpred = l.predict(X_test)
print 'linear SVM Accuracy is:' , accuracy_score(y_test, lpred)*100
print 'Testing Mean Squared error for linear SVM', mean_squared_error(y_test, lpred)

po = SVC(kernel='poly', degree=1)
po.fit(X_train, y_train)
popred = po.predict(X_test)
print 'Poly SVM Accuracy is:' , accuracy_score(y_test, popred)*100
print 'Testing Mean Squared error for Poly SVM', mean_squared_error(y_test, popred)

# NEURAL NETWORK ==============================================

from sklearn.neural_network import MLPClassifier


#NN Layer Variation
l_range = range(1, 20)
scoresnn = []
for l in l_range:
    nn = MLPClassifier(activation='logistic', hidden_layer_sizes=(l,), max_iter=800, random_state=100)
    nn.fit(X_train, y_train)  
    nn1_pred = nn.predict(X_test)
    scoresnn.append(accuracy_score(y_test,nn1_pred))
#plt.title("Testing accuracy for NN vs layers, GGG Dataset")
#plt.plot(l_range, scoresnn)
#plt.xlabel('Value of degree')
#plt.ylabel('Testing accuracy score')
#plt.show()


#NN with different activation functions
mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(14,), max_iter=1000, random_state=100)
mlp.fit(X_train, y_train)  
nn_pred = mlp.predict(X_test)
print 'Accuracy score for logistic Neural Network', accuracy_score(y_test,nn_pred)*100
print 'Mean Squared error for logistic NN ', mean_squared_error(y_test, nn_pred)

mlp2 = MLPClassifier(activation='identity', hidden_layer_sizes=(14,), max_iter=1000, random_state=100)
mlp2.fit(X_train, y_train)
nn2_pred = mlp2.predict(X_test)
print 'Accuracy score for identity neural network', accuracy_score(y_test,nn2_pred)*100
print 'Mean Squared error for id nn ', mean_squared_error(y_test, nn2_pred)


#LEARNIGN CURVE PLOT
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits= 25, test_size=0.2, random_state=0)


#title = "Decision Tree Learning Curve for GGG Dataset"
#p = plot_learning_curve(clf_gini1, title, X_train, y_train, cv=cv)
#p.show()

#title2 = "KNN Learning Curve for GGG Dataset"
#p2 = plot_learning_curve(knn, title2, X_train, y_train, cv=cv)
#p2.show()

#title3 = "SVM Learning Curve for GGG Dataset"
#p3 = plot_learning_curve(po, title3, X_train, y_train,cv=cv)
#p3.show()

title4 = "Neural Network Learning Curve for GGG Dataset"
p4 = plot_learning_curve(mlp, title4, X_train, y_train,cv=cv)
p4.show()

#title5 = "AdaBoosted Decision Tree Learning Curve, GGG Dataset"
#p5 = plot_learning_curve(boost, title5, X_train, y_train,cv=cv)
#p5.show()

