import numpy as np
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss
from sklearn.pipeline import Pipeline

def main():
    filename = 'train_final.csv'
    dataframe = pd.read_csv(filename)
    dummy_columns = ['workclass','occupation','marital.status','relationship','native.country','race','sex']
    new_dataframe = pd.get_dummies(dataframe, columns=dummy_columns, drop_first=False)
    
    feature_cols = ['age', 'education.num', 'capital.gain',
                    'capital.loss', 'hours.per.week', 'race_White',
                    'sex_Male', 'native.country_United-States',
                    'relationship_Husband', 'marital.status_Married-civ-spouse',
                    'marital.status_Never-married', 'relationship_Not-in-family',
                    'occupation_Prof-specialty', 'occupation_Craft-repair',
                    'occupation_Exec-managerial','workclass_Private',
                    'workclass_Self-emp-not-inc', 'workclass_Federal-gov',
                    'marital.status_Divorced', 'occupation_Sales',
                    'occupation_Transport-moving', 'occupation_Adm-clerical',
                    'occupation_Handlers-cleaners', 'occupation_Tech-support',
                    'relationship_Wife', 'relationship_Own-child', 'race_Black',
                    'race_Asian-Pac-Islander', 'sex_Female',
                    'native.country_Mexico']

    #not_feature_cols = ['ID', 'education', 'income>50K']
    #feature_cols = [column for column in new_dataframe.columns if column not in not_feature_cols]
    
    scaler = StandardScaler()

    X = scaler.fit_transform(new_dataframe.loc[:, feature_cols])
    y = dataframe['income>50K']

    over = SMOTE()
    under = NearMiss(version=1)

    #X,y = over.fit_resample(X,y)
    #X,y = under.fit_resample(X,y)
    #steps = [('o', over), ('u', under)]
    #pipeline = Pipeline(steps=steps)
    #X,y = pipeline.fit_resample(X,y)

    print(len(X)//2)

    split_num = len(X)//2
    X_train = X[:split_num]
    y_train = y[:split_num]

    X_test = X[split_num:]
    y_test = y[split_num:]

    #fs = SelectKBest(chi2, k=30).fit(X_train, y_train)

    #X_train = fs.transform(X_train)
    #X_test = fs.transform(X_test)
    #clf = BaggingClassifier(base_estimator=LogisticRegression(max_iter = 300), n_estimators = 15)
    clf = LogisticRegression(max_iter = 1000, multi_class="auto")
    clf.fit(X_train,y_train)

    score = accuracy_score(y_test, clf.predict(X_test))
    score_train = accuracy_score(y_train, clf.predict(X_train))
    print("Score on Train=", score_train)
    print("Score on Test=", score)
    print("Confusion Matrix=", confusion_matrix(y_test, clf.predict(X_test)))

    clf.fit(X, y)

    test_filename = 'test_final.csv'
    test = pd.read_csv(test_filename)
    new_test = pd.get_dummies(test, columns=dummy_columns, drop_first=False)
    X_new = scaler.fit_transform(new_test.loc[:, feature_cols])
    
    predictions = clf.predict(X_new)
    kaggle_data = pd.DataFrame({'ID':test.ID, 'Prediction':predictions}).set_index('ID')
    kaggle_data.to_csv('results.csv')
    
main()
