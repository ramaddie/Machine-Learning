import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import learning_curve

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn import tree


allData=pd.read_csv('input/character-predictions.csv')

allData=allData.drop(['S.No','isAlive','name','isAliveMother','isAliveFather','house','book1','book2','book3','book4','book5','spouse','DateoFdeath', 'mother', 'father', 'heir','alive', 'plod', 'title', 'culture', 'dateOfBirth', 'isAliveHeir', 'isAliveSpouse', 'isMarried', 'age', 'numDeadRelations', 'boolDeadRelations', 'popularity'],axis=1)

print(allData.head())
#print "Dataset Length: ", len(allData)
#print "Dataset Shape: ", allData.shape


#splitting features and target from dataset
X = allData.values[:, 1:]
Y = allData.values[:,0]

#manual split of 30% test and 70% training data
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

#PLOTTING DECISION TREE BASED ON DEPTH ============================================
d_range = range(1, 10)
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


#Decision Tree Classifier with criterion gini index and max_depth 4 (best)
clf_gini1 = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=1)
clf_gini1.fit(X_train, y_train)

#prediction
y_pred1 = clf_gini1.predict(X_test)
y1 = clf_gini1.predict(X_train)

#print y_pred1
print "Training accuracy for Decision Tree with max_depth 3 is: ", accuracy_score(y_train,y1)*100
print "Testing accuracy for Decision Tree with max_depth 3 is: ", accuracy_score(y_test,y_pred1)*100
print 'Training Mean Squared error for DT ', mean_squared_error(y_train, y1)
print 'Testing Mean Squared error for DT ', mean_squared_error(y_test, y_pred1)

def prune(decisiontree, min_samples_leaf):
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

#Pruning based on leaf nodes produces the same accuracy clf1
prune(clf_gini1, 3)
y_pred3 = clf_gini1.predict(X_test)
#print y_pred3

print "Accuracy for Decision Tree with max_depth 3 with pruning leaf nodes is the same: ", accuracy_score(y_test,y_pred3)*100

#Jon Snow prediction according to decision tree with 76% accuracy
js = clf_gini1.predict([[1,1,1,1]])
print "Prediction that jon snow will die is:", js
dani = clf_gini1.predict([[0,0,1,1]])
print "Prediction that dani will die is:", dani
cer = clf_gini1.predict([[1,0,1,0]])
print "Prediction that cersei will die is:", cer
tyr = clf_gini1.predict([[0,1,1,1]])
print "Prediction that tyrion will die is:", tyr

# AdaBoosting ===================================================

from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=4,
    learning_rate=1.5,
    algorithm="SAMME")
boost.fit(X_train, y_train)
b_pred = boost.predict(X_test)


#varied estimator
est_range = range(1,30)
scoresada = []
for e in est_range:
    b = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=e, learning_rate=1.5, algorithm="SAMME")
    b.fit(X_train, y_train)
    bpred = b.predict(X_test)
    scoresada.append(accuracy_score(y_test,bpred))
#plt.title("AdaBoosted Decision Tree accuracy vs. varying estimators")
#plt.plot(est_range, scoresada)
#plt.xlabel('Value of estimator')
#plt.ylabel('Testing accuracy score')
#plt.show()
    

print 'Accuracy for boosted DT', accuracy_score(y_test, b_pred)*100
print 'Mean Squared error for boosted DT ', mean_squared_error(y_test, b_pred)


#Jon Snow prediction according to decision tree with 76% accuracy

js = boost.predict([[1,1,1,1]])
print "Prediction that jon snow will die is:", js
dani = boost.predict([[0,0,1,1]])
print "Prediction that dani will die is:", dani
cer = boost.predict([[1,0,1,0]])
print "Prediction that cersei will die is:", cer
tyr = boost.predict([[0,1,1,1]])
print "Prediction that tyrion will die is:", tyr

#K Nearest Neighbor ==========================================
from sklearn.neighbors import KNeighborsClassifier

#PLOTTING BASED ON WHICH N = NEIGHBOR IS BEST
k_range = range(1, 25)
scores = []
for k in k_range:
    kn = KNeighborsClassifier(n_neighbors=k)
    kn.fit(X_train, y_train)
    y_kpred = kn.predict(X_test)
    scores.append(accuracy_score(y_test,y_kpred))
#plt.title("KNN accuracy vs. number of neighbors, GoT Dataset")
#plt.plot(k_range, scores)
#plt.xlabel('Value of k for KNN')
#plt.ylabel('Testing accuracy score')
#plt.show()

#'It looks like highest accuracy was achieved with n=19'
knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
y_tpred = knn.predict(X_train)
y_pred5 = knn.predict(X_test)
#print y_pred5
print "Traing Accuracy for KNN is :", accuracy_score(y_train, y_tpred)*100
print "Testing Accuracy for KNN with n=17: ", accuracy_score(y_test,y_pred5)*100
print 'Training Mean Squared error for KNN n=17 ', mean_squared_error(y_train, y_tpred)
print 'Testing Mean Squared error for KNN n=17 ', mean_squared_error(y_test, y_pred5)


#Prediction for jon snow with knn with accuracy of 77%
js = knn.predict([[1,1,1,1]])
print "Prediction that jon snow will die is:", js
dani = knn.predict([[0,0,1,1]])
print "Prediction that dani will die is:", dani
cer = knn.predict([[1,0,1,0]])
print "Prediction that cersei will die is:", cer
tyr = knn.predict([[0,1,1,1]])
print "Prediction that tyrion will die is:", tyr


#SVC ============================================
from sklearn.svm import SVC


#POLY KERNEL
deg_range = range(1, 20)
scores2 = []
for deg in deg_range:
    poly = SVC(kernel='poly', degree=deg)
    poly.fit(X_train, y_train)
    poly_pred = poly.predict(X_test)
    scores2.append(accuracy_score(y_test,poly_pred))

#plt.title("Polynomial SVM accuracy vs. degree used, GoT Dataset")
#plt.plot(deg_range, scores2)
#plt.xlabel('Value of degree')
#plt.ylabel('Testing accuracy score')
#plt.show()

li = SVC(kernel='linear', C=1)
li.fit(X_train, y_train)
lipred = li.predict(X_test)
print 'lin SVM Accuracy is:' , accuracy_score(y_test, lipred)*100
print 'Testing Mean Squared error for linear SVM', mean_squared_error(y_test, lipred)


po = SVC(kernel='poly', degree=3)
po.fit(X_train, y_train)
popred = po.predict(X_test)
print 'Poly SVM Accuracy is:' , accuracy_score(y_test, popred)*100
print 'Testing Mean Squared error for Poly SVM', mean_squared_error(y_test, popred)

#Prediction for jon snow with knn with accuracy of 77%
js = po.predict([[1,1,1,1]])
print "Prediction that jon snow will die is:", js
dani = po.predict([[0,0,1,1]])
print "Prediction that dani will die is:", dani
cer = po.predict([[1,0,1,0]])
print "Prediction that cersei will die is:", cer
tyr = po.predict([[0,1,1,1]])
print "Prediction that tyrion will die is:", tyr


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
mlp = MLPClassifier(activation='logistic', solver='adam', max_iter=260)
mlp.fit(X_train, y_train)  
nn_pred = mlp.predict(X_test)
print 'Accuracy score for logistic Neural Network', accuracy_score(y_test,nn_pred)*100
print 'Mean Squared error for logistic NN ', mean_squared_error(y_test, nn_pred)

mlp2 = MLPClassifier(activation='identity', solver='adam', max_iter=260)
mlp2.fit(X_train, y_train)
nn2_pred = mlp2.predict(X_test)
print 'Accuracy score for identity neural network', accuracy_score(y_test,nn2_pred)*100
print 'Mean Squared error for id nn ', mean_squared_error(y_test, nn2_pred)

js = mlp.predict([[1,1,1,1]])
print "Prediction that jon snow will die is:", js
dani = mlp.predict([[0,0,1,1]])
print "Prediction that dani will die is:", dani
cer = mlp.predict([[1,0,1,0]])
print "Prediction that cersei will die is:", cer
tyr = mlp.predict([[0,1,1,1]])
print "Prediction that tyrion will die is:", tyr


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


# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)


#title = "Decision Tree Learning Curve for GoT Dataset"
#p = plot_learning_curve(clf_gini1, title, X_train, y_train, cv=cv)
#p.show()

#title2 = "KNN Learning Curve for GoT Dataset"
#p2 = plot_learning_curve(knn, title2, X_train, y_train, cv=cv)
#p2.show()

#title3 = "SVM Learning Curve for GoT Dataset"
#p3 = plot_learning_curve(po, title3, X_train, y_train, cv=cv)
#p3.show()

title4 = "Neural Network Learning Curve for GoT Dataset"
p4 = plot_learning_curve(mlp, title4, X_train, y_train, cv=cv)
p4.show()

#title5 = "AdaBoosted Decision Tree Learning Curve, GoT Dataset"
#p5 = plot_learning_curve(boost, title5, X_train, y_train, cv=cv)
#p5.show()
