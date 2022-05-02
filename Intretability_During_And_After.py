import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.layers import LSTM

def handle_missing_values(data):

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(data)

    repaired_data =  imp.transform(data)
    data[:] = repaired_data
    print(data)
    return data


def get_columns_for_training(data):

    data.drop(columns=['observed_flow'],inplace=True)
    for col in data.columns:
        if len(data[col].unique()) == 1:
            data.drop(col, inplace=True, axis=1)
    return data

def select_features_by_filtering(X, Y):
    features = X.columns

    select = SelectKBest(score_func=f_regression, k=20)
    z = select.fit_transform(X, Y)
    filter = select.get_support()
    print(features[filter])
    return features[filter]
def plot_correlation_scores(X,Y):

    features = X.columns
    #print(features)
    #return
    k=len(features)
    select = SelectKBest(score_func=f_regression, k=k)

    z = select.fit_transform(X, Y)
    scores = select.scores_

    #print(scores)
    filter = select.get_support()
    feature_names = features[filter]

    for i in range(k):
        print(str(feature_names[i])+",\t"+ str(scores[i]) )
    #return
    print(scores)
    print(feature_names)

    feature_names = [x for _, x in sorted(zip(scores, feature_names), reverse=True)]
    scores = sorted(scores,reverse=True)

    for i in range(k):
        print(str(feature_names[i])+",\t"+ str(scores[i]) )

    #print(scores)
    #print(feature_names)
    fig = plt.figure(figsize=(15,6))

    # creating the bar plot
    plt.bar(feature_names[0:8], scores[0:8], color='maroon',
            width=0.25)

    plt.xlabel("Features")
    plt.ylabel("Pearson's Correlation Co-efficient")
    plt.title("Features vs Pearson's Correlation Co-efficient Scores")
    plt.show()
    return features[filter]

def filter_unnecessary_features(X_train, X_val, X_test, best_features):
    feature_list = []
    for feature in best_features:
        feature_list.append(str(feature))
    print(feature_list)
    datasets = [X_train, X_val, X_test]
    for data in datasets:
        for column in data:
            print(column)
            if column not in feature_list:
                #print(column)
                data.drop(column, inplace = True, axis = 1)
    print(X_test.columns)
    return X_train, X_val, X_test

def nse(predictions, targets):
    return (1-(np.sum((predictions-targets)**2)/np.sum((targets-np.mean(targets))**2)))

def get_NN_regression_model(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Dense(50, input_dim=20, activation="relu"))
    #model.add(Dropout(0.1))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(10, activation="relu"))
    #model.add(Dropout(0.05))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    model.fit(X_train, y_train, epochs=15)
    pred_train = model.predict(X_train)
    print("RMSE in training data: "+str(np.sqrt(mean_squared_error(y_train, pred_train))))

    pred = model.predict(X_val)
    print("RMSE in validation data: "+str(np.sqrt(mean_squared_error(y_val, pred))))
    return model


def get_std_ratio(pred, y_test):
    return np.std(pred)/np.std(y_test)


def run_NN_regression_model(model, X_test, y_test):
    pred = model.predict(X_test)
    pred = np.reshape(pred, len(pred))
    print(pred)
    print(y_test)
    nse_value = nse(pred, y_test)
    std_ratio = get_std_ratio(pred, y_test)
    print("NSE in test data: " + str(nse_value))
    print("STD Ratio in test data: " + str(std_ratio))

    print("RMSE in test data: " + str(np.sqrt(mean_squared_error(y_test, pred))))

def get_LSTM_model(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(LSTM(20, input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=10, batch_size=10)
    pred_train = model.predict(X_train)
    print("RMSE in training data: " + str(np.sqrt(mean_squared_error(y_train, pred_train))))

    pred = model.predict(X_val)
    print("RMSE in validation data: " + str(np.sqrt(mean_squared_error(y_val, pred))))
    return model

data = pd.read_csv("data/interpreter_data.csv", index_col= 'gauge_id')
y= data['observed_flow']

X = get_columns_for_training(data)


X = handle_missing_values(data)
# split into-train test and validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)

best_features = select_features_by_filtering(X_train, y_train)





X_train, X_val, X_test = filter_unnecessary_features(X_train, X_val, X_test, best_features)





#model = get_LSTM_model(X_train, y_train, X_val, y_val)
#run_NN_regression_model(model, X_test, y_test)

model = get_NN_regression_model(X_train, y_train, X_val, y_val)
run_NN_regression_model(model, X_test, y_test)


print(best_features)
#plot_correlation_scores(X_train, y_train)


#print(data)