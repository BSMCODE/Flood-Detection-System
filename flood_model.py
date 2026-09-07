import numpy as np
import pandas as pd
import matplotlib as plt

# logistic regression model code

# sigmoid function
def sigmoid(z): 
    return 1/(1 +np.exp(-z))

def logistic_regression_train(X, y, learning_rate=0.01, iterations=1000): 
    m, n = X.shape #n is num of features (eg. rain_lag1, rain_lag2 etc.)
    weights = np.zeros(n) #all weights initialise to 0, weights = b1 ....
    bias = 0 #final intercept variable

    for i in range(iterations):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model) #calculates predicted value for each row using sigmoid(X x w + b)

        err = predictions-y

        dw = (1 / m) * np.dot(X.T, err) #calculates how much to change weight dw and bias db based on avg error
        db = (1 / m) * np.sum(err)

        weights -= learning_rate * dw #update valus of weights, bias
        bias -= learning_rate * db #b0

    return weights, bias

def logistic_regression_predict(X, weights, bias, threshold=0.5):
    probs = sigmoid(np.dot(X, weights) + bias) # calculates how likely each row is a flood
    return (probs >= threshold).astype(int) # if prob > 0.5, predicts flood (outputs '1')
