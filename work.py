# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import metrics, linear_model, svm, tree
import xgboost as xgb
from keras.layers import Input,Embedding,Bidirectional,LSTM,Dropout,TimeDistributed,Dense,Activation,Reshape
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model
from keras.models import Sequential
import lightgbm as lgb



def main():
    print("# Loading data...")
    # The training data is used to train your model how to predict the targets.
    train = pd.read_csv('pzo_training_data.csv', header=0)
    # The test data is the data that pzo uses to evaluate your model.
    test = pd.read_csv('pzo_test_data.csv', header=0)
    
    # The test data contains validation data, test data and live data.
    # Validation is used to test your model locally so we separate that.
    validation = test[test['data_type'] == 'validation']
    
    # There are multiple targets in the training data which you can choose to model using the features.
    # pzo does not say what the features mean but that's fine; we can still build a model.
    # Here we select the bernie_target.
    
    
    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(train) if "feature" in f]
    X = train[features]
    Y = train['target']
    x_prediction = validation[features]
    print('#train target')
    print(sum(train['target'])/len(train['target']))
    print('#validation target')
    print(sum(validation['target'])/len(validation['target']))
    
    #logistic regression
    # This is your model that will learn to predict this target.
    model_logistic = linear_model.LogisticRegression(n_jobs=-1)
    print("# Training...")
    # Your model is trained on train_bernie
    model_logistic.fit(X, Y)
    
    print("# Predicting...")
    # Based on the model we can predict the probability of each row being
    # a bernie_target in the validation data.
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction_logistic = model_logistic.predict_proba(x_prediction)
    probabilities_logistic = y_prediction_logistic[:, 1]
    print("- probabilities:", probabilities_logistic[1:6])
    
    correct_logistic = [
        round(x) == y
        for (x, y) in zip(probabilities_logistic, validation['target'])
    ]
    print("- accuracy: ", sum(correct_logistic) / float(validation.shape[0]))
    accuracy_logistic = sum(correct_logistic) / float(validation.shape[0])
    
    
    # pzo measures models on logloss instead of accuracy. The lower the logloss the better.
    # Our validation logloss isn't very good.
    print("- validation logloss:",
          metrics.log_loss(validation['target'], probabilities_logistic))
    logloss_logistic = metrics.log_loss(validation['target'], probabilities_logistic)
    
    # To submit predictions from your model to pzo, predict on the entire test data.
    x_prediction = test[features]
    y_prediction_logistic = model_logistic.predict_proba(x_prediction)
    results_logistic = y_prediction_logistic[:, 1]
    
    
    #SVM
    x_prediction = validation[features]
    model_svm = svm.SVC(probability = True)
    
    print("# Training...")
    # Your model is trained on train_bernie
    model_svm.fit(X, Y)
    
    print("# Predicting...")
    # Based on the model we can predict the probability of each row being
    # a bernie_target in the validation data.
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction_svm = model_svm.predict_proba(x_prediction)
    probabilities_svm = y_prediction_svm[:, 1]
    print("- probabilities:", probabilities_svm[1:6])
    
    correct_svm = [
        round(x) == y
        for (x, y) in zip(probabilities_svm, validation['target'])
    ]
    print("- accuracy: ", sum(correct_svm) / float(validation.shape[0]))
    accuracy_svm = sum(correct_svm) / float(validation.shape[0])
    
    
    # pzo measures models on logloss instead of accuracy. The lower the logloss the better.
    # Our validation logloss isn't very good.
    print("- validation logloss:",
          metrics.log_loss(validation['target'], probabilities_svm))
    logloss_svm = metrics.log_loss(validation['target'], probabilities_svm)
    
    # To submit predictions from your model to pzo, predict on the entire test data.
    x_prediction = test[features]
    y_prediction_svm = model_svm.predict_proba(x_prediction)
    results_svm = y_prediction_svm[:, 1]
    
    
    
    x_prediction = validation[features]
    model_gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31,learning_rate=0.05,n_estimators=20)
    model_gbm.fit(X, Y,eval_set=[(validation[features], validation['target'])],eval_metric='l1',early_stopping_rounds=5)
    
    print("# Predicting...")
    # Based on the model we can predict the probability of each row being
    # a bernie_target in the validation data.
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction_gbm = model_gbm.predict(x_prediction)
    probabilities_gbm = y_prediction_gbm
    print("- probabilities:", probabilities_gbm[1:6])
    
    correct_gbm = [
        round(x) == y
        for (x, y) in zip(probabilities_gbm, validation['target'])
    ]
    print("- accuracy: ", sum(correct_gbm) / float(validation.shape[0]))
    accuracy_svm = sum(correct_gbm) / float(validation.shape[0])
    
    
    # pzo measures models on logloss instead of accuracy. The lower the logloss the better.
    # Our validation logloss isn't very good.
    print("- validation logloss:",
          metrics.log_loss(validation['target'], probabilities_gbm))
    logloss_gbm = metrics.log_loss(validation['target'], probabilities_gbm)
    
    # To submit predictions from your model to pzo, predict on the entire test data.
    x_prediction = test[features]
    y_prediction_gbm = model_gbm.predict(x_prediction)
    results_gbm = y_prediction_gbm
    
    
    
    #xgb
    import xgboost as xgb
    data_train = xgb.DMatrix(X, Y)
    data_test = xgb.DMatrix(test[features], test['target'])
    data_valtidation = xgb.DMatrix(validation[features],validation['target'])
    param = {'max_depth': 5, 'eta': 1, 'objective': 'binary:logistic'}
    watchlist = [(data_test, 'test'), (data_train, 'train')]
    n_round = 3
    model_xgb = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    probabilities_xgb = model_xgb.predict(data_valtidation)
    
    correct_xgb = [
        round(x) == y
        for (x, y) in zip(probabilities_xgb, validation['target'])
    ]
    print("- accuracy: ", sum(correct_xgb) / float(validation.shape[0]))
    accuracy_xgb = sum(correct_xgb) / float(validation.shape[0])
    print("- validation logloss:",
          metrics.log_loss(validation['target'], probabilities_xgb))
    logloss_xgb = metrics.log_loss(validation['target'], probabilities_xgb)
    
    y_prediction_xgb = model_xgb.predict(data_test)
    results_xgb = y_prediction_xgb
    
    
    
    #LSTM
    from keras.layers import Input,Embedding,Bidirectional,LSTM,Dropout,TimeDistributed,Dense,Activation,Reshape
    from keras.callbacks import EarlyStopping
    from keras.utils import to_categorical
    from keras.models import Model
    from keras.models import Sequential
    Y = train['target']
    Y = to_categorical(Y)
    val_x = validation[features]
    val_y = to_categorical(validation['target'])
    max_length = 50
    model = Sequential()
    #model.add(Reshape((max_length,1),input_shape=(max_length,)))
    model.add(Dense(64,activation = 'relu',input_shape = (max_length,)))
    model.add(Dense(2,activation = 'softmax'))
    model.compile('adam',loss='binary_crossentropy',metrics=['accuracy'])  
    early_stop = EarlyStopping(monitor = 'val_loss',mode = 'min',patience = 3)
    model.fit(X,Y,batch_size = 20,epochs = 20,validation_data = [val_x, val_y],callbacks = [early_stop],verbose = 2)
    
    y_prediction_ann = model.predict(val_x)
    probabilities_ann = y_prediction_ann[:,1]
    
    correct_ann = [
        round(x) == y
        for (x, y) in zip(probabilities_ann, validation['target'])
    ]
    print("- accuracy: ", sum(correct_ann) / float(validation.shape[0]))
    accuracy_ann = sum(correct_ann) / float(validation.shape[0])
    
    
    # pzo measures models on logloss instead of accuracy. The lower the logloss the better.
    # Our validation logloss isn't very good.
    print("- validation logloss:",
          metrics.log_loss(validation['target'], probabilities_ann))
    logloss_ann = metrics.log_loss(validation['target'], probabilities_ann)
    
    # To submit predictions from your model to pzo, predict on the entire test data.
    x_prediction = test[features]
    y_prediction_ann = model.predict_proba(x_prediction)
    results_ann = y_prediction_ann[:, 1]






    print("# Creating submission...")
    results_df = pd.DataFrame(data={'logistic': results_logistic,'svm':results_svm,
                                    'xgb':results_xgb,'ann':results_ann,'gbm':results_gbm})
    print("# Writing predictions to submissions.csv...")
    results_df.to_csv("submission.csv", index=False)
    
    
if __name__ == '__main__':
    main()