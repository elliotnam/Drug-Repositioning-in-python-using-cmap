import pandas as pd
from sklearn import model_selection
from sklearn import cross_validation,metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy
from sklearn.feature_selection import RFE
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import cPickle

url = "/home/elliotnam/project/cmap/DrugFingerPrint_final.csv"



dataframe = pd.read_csv(url)
#dataframe.drop(dataframe.index[[0]])
print(dataframe)
#dataframe2 = dataframe[1:]

# only chem infos
#colNames = dataframe.columns[4:170]
#array = dataframe.values
#print('now array values..............')
#print(array[0,4:170])
#X = array[:,4:170]
#X = X.astype(numpy.int64)

# only gen infos
#colNames = dataframe.columns[170:]
#array = dataframe.values
#print('now array values..............')
#print(array[0,170:])
##X = array[:,170:]
#X = X.astype(numpy.int64)
#X = X.astype(numpy.int64)

# gen _ infos
colNames = dataframe.columns[4:]
array = dataframe.values
print('now array values..............')
print(array[0,4:])
X = array[:,4:]
X = X.astype(numpy.int64)



Y = array[:,2]
#print(Y[0])
#print(Y[1])
#Y = label_binarize(Y, classes=['1','2'])
num_folds = 5
num_instances = len(X)
seed = 7
#print(Y)
Y = Y.astype(numpy.int64)
#Y = int(Y)# label_binarize(Y, classes=[0,1])
#print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2,
                                                    random_state=0)
kfold = model_selection.KFold(n_splits= num_folds, random_state=seed)

def runLogisticRegression():

    model = LogisticRegression()
    results = model_selection.cross_val_score(model, X, Y.ravel(), cv=kfold,n_jobs=8)
    i = 0

    print(results.mean())
    with open("/home/elliotnam/project/cmap/model/lrmodel_result.pkl","wb") as fid:
        cPickle.dump(results,fid)


    with open("/home/elliotnam/project/cmap/model/lrmodel.pkl", "wb") as fid:
        cPickle.dump(model,fid)


def runLinearDiscriment():
    model = LinearDiscriminantAnalysis()
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=4)
    print(results.mean())


    with open("/home/elliotnam/project/cmap/model/ldmodel.pkl", "wb") as fid:
        cPickle.dump(results, fid)


def runKNNClassification():
    kfold = model_selection.KFold(n_splits=num_folds,
                                   random_state=seed)
    model = KNeighborsClassifier()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())

    with open("/home/elliotnam/project/cmap/model/knnmodel_result.pkl","wb") as fid:
        cPickle.dump(results,fid)

    with open("/home/elliotnam/project/cmap/model/knnmodel.pkl","wb") as fid:
        cPickle.dump(model,fid)

def runNaiveBayes():
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    model = GaussianNB()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())


    with open("/home/elliotnam/project/cmap/model/naivebayesmodel_result.pkl", "wb") as fid:
        cPickle.dump(results, fid)


    with open("/home/elliotnam/project/cmap/model/naivebayesmodel.pkl", "wb") as fid:
        cPickle.dump(model, fid)

def runDecisionTree():
    scoring = 'accuracy'
    model = DecisionTreeClassifier()
    kfold = model_selection.KFold(n_splits =num_folds, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y.ravel(), cv=kfold, scoring=scoring, n_jobs=8)

    with open("/home/elliotnam/project/cmap/model/decisiontreemodel_result.pkl","wb") as fid:
        cPickle.dump(results,fid)


    with open("/home/elliotnam/project/cmap/model/decisiontreemodel.pkl","wb") as fid:
        cPickle.dump(model,fid)

    #results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=-1)
    print(results.mean())

'''
    i =0
    for train_index, test_index in kfold:
        if i == 2:
            model.fit(X[train_index],Y[train_index].ravel())
            predictions = model.predict(X[test_index])
            dtrain_predprob = model.predict_proba(X[train_index])[:,1]
            feat_imp = pd.Series(model.feature_importances_,colNames).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Decision Free Feature Importance Score')
            plt.show()
        i += 1
'''

def runSVM():
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    model = SVC()
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())

    with open("/home/elliotnam/project/cmap/model/svmmodel_result.pkl","wb") as fid:
        cPickle.dump(results,fid)


    with open("/home/elliotnam/project/cmap/model/svmmodel.pkl","wb") as fid:
        cPickle.dump(model,fid)

def tunningLogisticRegression():
    #estimater = LogisticRegression()
    research =GridSearchCV(cv=None,
                 estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
                                              penalty='l2', tol=0.0001),
                 param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})



    research.fit(X, Y)
    print(research.best_score_)
    print(research.best_params_)

#def tunningGradientBoost():


def testGradeintBoost():
    param_test1 = {'n_estimators': range(20, 81, 10)}
    kfold = model_selection.KFold(n_splits=2, random_state=seed)
    gsearch1 = GridSearchCV(
        estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, min_samples_leaf=50, max_depth=8,
                                             max_features='sqrt', subsample=0.8, random_state=10),
        param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

    for i, (train, test) in enumerate(kfold):
#        gsearch1.fit(X[train], Y[train].ravel())
#        print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)
        print()

    param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
    gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30, max_features='sqrt', subsample=0.8, random_state=10),
    param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

    for i, (train, test) in enumerate(kfold):
#        gsearch2.fit(X[train], Y[train].ravel())
#        print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
        print()

    param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
    gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,max_depth=5,max_features='sqrt', subsample=0.8, random_state=10),
    param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    for i, (train, test) in enumerate(kfold):
#        gsearch3.fit(X[train], Y[train].ravel())
#        print(gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
        print()
    param_test4 = {'max_features':range(6,10,2)}
    gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,max_depth=5, min_samples_split=1600, min_samples_leaf=50, subsample=0.8, random_state=7),
    param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    for i, (train, test) in enumerate(kfold):
        #gsearch4.fit(X[train], Y[train].ravel())
        print(gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_)

    param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
    gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=30,max_depth=5,min_samples_split=1600, min_samples_leaf=50, subsample=0.8, random_state=7,max_features=11),
    param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

    for i, (train, test) in enumerate(kfold):
#        gsearch5.fit(X[train], Y[train].ravel())
#        print(gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_)
        print()

    predictors = [x for x in X[0]]
    gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=30,max_depth=5, min_samples_split=1600,min_samples_leaf=50, subsample=0.8, random_state=7)
    modelfit(gbm_tuned_1, X_train, predictors)

def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    rcParams['figure.figsize'] = 12,4
    target = 'Disbursed'
    IDcol = 'ID'
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = model_selection.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds,
                                                    scoring='roc_auc')

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
            np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


def runCompareAlgorithms():
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC()))

    num_trees = 100
    models.append(('GBC',GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)))

    num_trees = 30
    models.append(('ABC',AdaBoostClassifier(n_estimators=num_trees, random_state=seed)))

    num_trees = 100
    max_features = 13
    models.append(('RFC',RandomForestClassifier(n_estimators=num_trees, max_features=max_features)))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X, Y.ravel(), cv=kfold, scoring=scoring,n_jobs=-1)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle("Algorithm Comparison")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def runBaggedDecisionTree():
    kfold = model_selection.KFold( n_splits=num_folds, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())

    with open("/home/elliotnam/project/cmap/model/baggeddecisionmodel_result.pkl","wb") as fid:
        cPickle.dump(results,fid)

    with open("/home/elliotnam/project/cmap/model/baggeddecisionmodel.pkl","wb") as fid:
        cPickle.dump(model,fid)

def runRandomForest():
    num_trees = 100
    max_features = 3
    kfold = model_selection.KFold( n_splits=num_folds, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())


    with open("/home/elliotnam/project/cmap/model/randomforestmodel_result.pkl", "wb") as fid:
        cPickle.dump(results, fid)

    with open("/home/elliotnam/project/cmap/model/randomforestmodel.pkl", "wb") as fid:
        cPickle.dump(model, fid)

def runExtraTrees():
    num_trees = 100
    max_features = 12
    kfold = model_selection.KFold( n_splits=num_folds, random_state=seed)
    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())
    with open("/home/elliotnam/project/cmap/model/extratreemodel_result.pkl","wb") as fid:
        cPickle.dump(results,fid)


    with open("/home/elliotnam/project/cmap/model/extratreemodel.pkl","wb") as fid:
        cPickle.dump(model,fid)

def runadaBust():
    num_trees = 30
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())

    with open("/home/elliotnam/project/cmap/model/adabustmodel_result.pkl","wb") as fid:
        cPickle.dump(results,fid)

    with open("/home/elliotnam/project/cmap/model/adabustmodel.pkl","wb") as fid:
        cPickle.dump(model,fid)


def viewGradientBustFeatures():
    num_trees = 100
    print('run gradeint')
    kfold = model_selection.KFold( n_splits=num_folds, random_state=seed)
    model = GradientBoostingClassifier(learning_rate=0.005,n_estimators=num_trees, random_state=seed,max_depth=5, min_samples_split=1600,min_samples_leaf=50, subsample=0.8)
    i =0
    for train_index, test_index in kfold:
        if i == 2:
            model.fit(X[train_index], Y[train_index].ravel())
            predictions = model.predict(X[test_index])
            dtrain_predprob = model.predict_proba(X[train_index])[:, 1]
            feat_imp = pd.Series(model.feature_importances_, colNames).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Gradient Bust Feature Importance Score')
            plt.show()
        i += 1

def runGradientBust():
    num_trees = 100
    print('run gradeint')
    kfold = model_selection.KFold( n_splits=num_folds, random_state=seed)
    model = GradientBoostingClassifier(learning_rate=0.005,n_estimators=num_trees, random_state=seed,max_depth=5, min_samples_split=1600,min_samples_leaf=50, subsample=0.8)
    results = model_selection.cross_val_score(model, X, Y.ravel(), cv=kfold,scoring='roc_auc',n_jobs=8)
    print("result:...........")
    print(results)

    for train, test in kfold.split(X):
        print(X[1])
        model.fit(X[train], Y[train])
        # print(alg.grid_scores_, alg.best_params_, alg.best_score_,alg.feature_importances_)

        print()
        print(model.feature_importances_)
        feat_imp = pd.Series(model.feature_importances_, colNames).sort_values(ascending=False)
        print(feat_imp)
        feat_imp2 = feat_imp[feat_imp > 0]
        feat_imp2.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
        print()

def runVotingLearning():
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    # create the sub models
    estimators = []
    model1 = LogisticRegression()
    estimators.append(("logistic", model1))
    model2 = DecisionTreeClassifier()
    estimators.append(("cart", model2))
    #model3 = SVC()
    model3=GaussianNB()
    estimators.append(("basyen", model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold,n_jobs=8)
    print(results.mean())
    with open("/home/elliotnam/project/cmap/model/votingmodel.pkl","wb") as fid:
        cPickle.dump(results,fid)


def runTunning():
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    # create the sub models
    #estimators = []
    model1 = LogisticRegression()
    #estimators.append(("logistic", model1))
    model2 = DecisionTreeClassifier()
    #estimators.append(("cart", model2))
    #model3 = SVC()
    model3=GaussianNB()
    #estimators.append(("basyen", model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    param_grid = {"alpha": uniform()}
    iterations = 100
    research = RandomizedSearchCV(estimator=ensemble,param_distributions=param_grid,n_iter=iterations,random_state=seed)
    research.fit(X,Y)
    print(research.best_score_)
    print(research.best_estimator_.alpha)

def runUnivariateSelection():
    test = SelectKBest(score_func=chi2, k=7)
    fit = test.fit(X, Y)
    # summarize scores
    numpy.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    # summarize selected features
    print(features[0:7, :])


def runRecursiveFeaureDelete():
    model = LogisticRegression()
    rfe = RFE(model, 7)
    fit = rfe.fit(X, Y)
    print("Num Features: %d") % fit.n_features_
    print("Selected Features: %s") % fit.support_
    print("Feature Ranking: %s") % fit.ranking_


def runExtraTreeFeatureImportance():
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)


def viewImportanceFeatures(fileName):
    num_folds =2
    num_trees = 100
    kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
    alg = GradientBoostingClassifier(learning_rate=0.005, n_estimators=num_trees, random_state=seed, max_depth=5,
                                       min_samples_split=1600, min_samples_leaf=50, subsample=0.8)
    #with open(fileName,"rb") as fid:
    #    alg = cPickle.load(fid)

    for train, test in kfold.split(X):
        print(X[1])
        alg.fit(X[train], Y[train])
        #print(alg.grid_scores_, alg.best_params_, alg.best_score_,alg.feature_importances_)
        print()
        print(alg.feature_importances_)
        feat_imp = pd.Series(alg.feature_importances_, colNames).sort_values(ascending=False)
        print(feat_imp)
        feat_imp2 = feat_imp[feat_imp > 0]
        feat_imp2.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
        print()

#runLogisticRegression()
#runLinearDiscriment()
#runKNNClassification()
#runNaiveBayes()
#runDecisionTree()
runSVM()
#runCompareAlgorithms()
##runBaggedDecisionTree()
#runRandomForest()
#runExtraTrees()
#runadaBust()
#runGradientBust()
#runVotingLearning()
#runTunning()
#runUnivariateSelection()
#runRecursiveFeaureDelete()
#runExtraTreeFeatureImportance()

#tunningLogisticRegression()
#testGradeintBoost()

#print("gradient bust model")
#viewImportanceFeatures("/home/elliotnam/project/cmap/model/gradientbustmodel.pkl")

#print("svm model")
#viewImportanceFeatures("/home/elliotnam/project/cmap/model/svmmodel.pkl")

#print("adabust")
#viewImportanceFeatures("/home/elliotnam/project/cmap/model/adabustmodel.pkl")
#viewGradientBustFeatures()