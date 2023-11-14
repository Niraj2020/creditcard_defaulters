from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: Neeraj
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.gnb = GaussianNB()
        self.xgb = XGBClassifier(objective='binary:logistic',n_jobs=-1)
        self.clf = RandomForestClassifier()
        self.knn = KNeighborsClassifier()
        #self.dtc = DecisionTreeClassifier()
        #self.lr  = LogisticRegression()
        self.sv  = SVC()

    def get_best_params_for_naive_bayes(self,train_x,train_y):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the Naive Bayes's Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Neeraj
        Version: 1.0
        Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.gnb, param_grid=self.param_grid, cv=3,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.var_smoothing = self.grid.best_params_['var_smoothing']


            #creating a new model with the best parameters
            self.gnb = GaussianNB(var_smoothing=self.var_smoothing)
            # training the mew model
            self.gnb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Naive Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.gnb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Naive Bayes Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Neeraj
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                "n_estimators": [50,100, 130],
                               "max_depth": range(3, 11, 1),
    "random_state":[0,50,100]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=2,n_jobs=-1)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.random_state = self.grid.best_params_['random_state']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(random_state=self.random_state, max_depth=self.max_depth,n_estimators= self.n_estimators, n_jobs=-1 )
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Neeraj
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5, verbose=3)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_KNN(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_KNN
                                                Description: get the parameters for KNN Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: Neeraj
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_Ensembled_KNN method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_knn = {
                'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
                'leaf_size' : [10,17,24,28,30,35],
                'n_neighbors':[4,5,8,10,11],
                'p':[1,2]
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.knn, self.param_grid_knn, verbose=3,
                                     cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.algorithm = self.grid.best_params_['algorithm']
            self.leaf_size = self.grid.best_params_['leaf_size']
            self.n_neighbors = self.grid.best_params_['n_neighbors']
            self.p  = self.grid.best_params_['p']

            # creating a new model with the best parameters
            self.knn = KNeighborsClassifier(algorithm=self.algorithm, leaf_size=self.leaf_size, n_neighbors=self.n_neighbors,p=self.p,n_jobs=-1)
            # training the mew model
            self.knn.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'KNN best params: ' + str(
                                       self.grid.best_params_) + '. Exited the KNN method of the Model_Finder class')
            return self.knn
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'knn Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_SVC(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_SVC
                                                Description: get the parameters for SVC Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Written By: Neeraj
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_DecisionTreeClassifier method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_SVC = {
                'C': [0.1,1, 10, 100],
                'gamma': [1,0.1,0.01,0.001],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.sv,self.param_grid,refit=True,verbose=2)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.C = self.grid.best_params_['C']
            self.gamma = self.grid.best_params_['gamma']
            self.kernel = self.grid.best_params_['kernel']

            # creating a new model with the best parameters
            self.sv = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel,n_jobs=-1)
            # training the mew model
            self.sv.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVC best params: ' + str(
                                       self.grid.best_params_) + '. Exited the SVC method of the Model_Finder class')
            return self.sv
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'SVC Parameter tuning  failed. Exited the knn method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Neeraj
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for naive_bayes
            self.naive_bayes=self.get_best_params_for_naive_bayes(train_x,train_y)
            self.prediction_naive_bayes=self.naive_bayes.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.naive_bayes_score = accuracy_score(test_y,self.prediction_naive_bayes)
                self.logger_object.log(self.file_object, 'Accuracy for NB:' + str(self.naive_bayes_score))
            else:
                self.naive_bayes_score = roc_auc_score(test_y, self.prediction_naive_bayes) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.naive_bayes_score))


            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.random_forest_score))

            # create best model for knn
            self.knn=self.get_best_params_for_KNN(train_x,train_y)
            self.prediction_knn=self.knn.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.knn_score = accuracy_score(test_y,self.prediction_knn)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.knn_score))
            else:
                self.knn_score = roc_auc_score(test_y, self.prediction_knn) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.knn_score))


            # create best model for svc
                self.sv = self.get_best_params_for_SVC(train_x, train_y)
                self.prediction_sv = self.knn.predict(test_x)  # prediction using the Random Forest Algorithm

                if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    self.sv_score = accuracy_score(test_y, self.prediction_sv)
                    self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.sv_score))
                else:
                    self.sv_score = roc_auc_score(test_y, self.prediction_sv)  # AUC for Random Forest
                    self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.sv_score))

            #comparing the two models
            if(self.random_forest_score  <  self.xgboost_score):
                return 'XGBoost',self.xgboost
            else:
                return 'RandomForest',self.random_forest_score

          #  results = pandas.DataFrame({
           #     'Model': ['naive bayes', 'xgboost', 'KNN', 'SVR',
            #              'Random Forest'],
             #   'Score': [self.naive_bayes_score,self.xgboost_score,self.knn_score,self.sv_score,self.random_forest_score]})
            #a = results[results['Score'] == max(results['Score'])]


        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()