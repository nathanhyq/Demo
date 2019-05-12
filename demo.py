#!/usr/bin/env python
"""
Example classifier on Point Zero One data using a logistic regression classifier.
To get started, install the required packages
"""

import pandas as pd
import numpy as np
from sklearn import metrics, linear_model


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


    # This is your model that will learn to predict this target.
    model = linear_model.LogisticRegression(n_jobs=-1)
    print("# Training...")
    # Your model is trained on train_bernie
    model.fit(X, Y)

    print("# Predicting...")
    # Based on the model we can predict the probability of each row being
    # a bernie_target in the validation data.
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(x_prediction)
    probabilities = y_prediction[:, 1]
    print("- probabilities:", probabilities[1:6])

    correct = [
        round(x) == y
        for (x, y) in zip(probabilities, validation['target'])
    ]
    print("- accuracy: ", sum(correct) / float(validation.shape[0]))


    # pzo measures models on logloss instead of accuracy. The lower the logloss the better.
    # Our validation logloss isn't very good.
    print("- validation logloss:",
          metrics.log_loss(validation['target'], probabilities))

    # To submit predictions from your model to pzo, predict on the entire test data.
    x_prediction = test[features]
    y_prediction = model.predict_proba(x_prediction)
    results = y_prediction[:, 1]

    print("# Creating submission...")
    # Create your submission
    results_df = pd.DataFrame(data={'probability': results})

    print("# Writing predictions to submissions.csv...")
    # Save the predictions out to a CSV file.
    results_df.to_csv("submission.csv", index=False)



if __name__ == '__main__':
    main()
