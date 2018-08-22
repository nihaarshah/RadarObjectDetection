# Cross validate hyperparameter tuning
# USe ipython
import sys
sys.path.insert(0, '/Users/nihaar/Documents/4yp/4yp/code/helpers/')
import load_data_rotated
x_train, y_train = load_data_rotated.load_train_data_SVM_rotated()

from sklearn import datasets, svm, metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

param_grid = [{'C': [1, 10 , .5, .05], 'kernel': ['linear']}]
classifier = svm.SVC(kernel='linear',probability=True)
grid_search = GridSearchCV(classifier,param_grid,cv=5,scoring='neg_log_loss')
grid_search.fit(x_train,y_train)
grid_search.best_params_
grid_search.best_estimator_

cvres = grid_search.cv_results_
for mean_score, std_dev, params in zip(cvres["mean_test_score"], cvres["std_test_score"],cvres["params"]):
	print(mean_score, std_dev, params)


from sklearn.externals import joblib
    ...: joblib.dump(grid_search, 'filename.pkl')
# to load the object later
    gr_ob = joblib.load('filename.pkl')