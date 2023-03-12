import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import streamlit as slt

class LogisticRegression():

    def __init__(self, learning_rate, no_of_iterations):

        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    # fit function to train the model with dataset.
    def fit(self, X, Y):
        # X -> matrics with all columns (exept outcomes)
        # Y -> outcomes column
        # m -> (number of rows) datapoints in dataset
        # n -> (number of columns) input features in dataset.
        self.m, self.n = X.shape
        # initiating weight and bias value.
        # weight will always be in matrics of input features.
        self.w = np.zeros(self.n)
        self.b = 0

        self.X = X
        self.Y = Y

        # implement Gradient methods for optimization.
        for i in range(self.no_of_iterations):
            self.update_weight()

    def update_weight(self):
        # sigmoid function.
        # formulae -> 1 / (1 + exp(-z)) where z = wX + b
        Y_hat = 1 / (1 + np.exp(-(self.X.dot(self.w) + self.b)))

        # derivatives.
        # we will  take transpose of X. becoz both matrix columns
        #  should be same for multiply. ex -> 2 x 2 not 2 x 3.
        dw = (1/self.m) * np.dot(self.X.T, (Y_hat - self.Y))
        db = (1/self.m) * np.sum(Y_hat - self.Y)

    # updating Gradient methods.
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

      # predict will return either person is diabetic or not.
    def predict(self, X):

        Y_pred = 1 / (1 + np.exp(-(X.dot(self.w) + self.b)))
        # where works like if else block. if(condition) ? statment1 : statment2
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        return Y_pred


# model = LogisticRegression(learning_rate=0.01,no_of_iterations=1000)

dataset = pd.read_csv('./diabetes.csv')
##print(dataset.head())

# print rows and columns
dataset.shape
# prints statistical analysis.
dataset.describe()

# print how many are diabetic and non diabetic.
# 0 -> non-diabetic
# 1 -> Diabetic
dataset['Outcome'].value_counts()

dataset.groupby('Outcome').mean()
# person with diabetic will have more glucose level and insulin

# seperating data  and labels
features = dataset.drop(columns='Outcome', axis=1)
target = dataset['Outcome']

# print(features)
#print(target)

#standardizing the data.
scaler = StandardScaler()
scaler.fit(features)

standarized_data = scaler.transform(features)
#print(standarized_data)

features = standarized_data

#sending data for splitting into training and testing.
X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.2, random_state=2)
#print(X_train.shape, X_test.shape)

# Training the model
classifier = LogisticRegression(learning_rate=0.01, no_of_iterations=1000)
# training the support vector machine classifier.
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
# Y-train -> true value , X_train_prediction -> predicte value.
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

#print('Accuracy score of training data : ', training_data_accuracy)

X_test_predictions = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_predictions)

#print('Accuracy of Testing data : ', testing_data_accuracy)

# loading the saved model
load_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetic_prediction(input_data):

    # changing input data to numpy array.
    input_data_as_array = np.asarray(input_data)
    # reshape array for one instance.
    input_data_reshaped = input_data_as_array.reshape(1, -1)

    # standardize the data.
    std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = load_model.predict(std_data)
    #print(prediction)

    if (prediction[0] == 0):
       return 'person is not diabetic'
    else:
       return 'Person is Diabetic'
    
def main():

    slt.header('Diabetic prediction App.')

    preganancies = slt.text_input('Number of Preganancies')
    glucose = slt.text_input('Glucose')
    BloodPressure = slt.text_input('Blood Pressure value')
    SkinThickness = slt.text_input('Skin thickness value')
    Insulin = slt.text_input('Insulin value')
    BMI = slt.text_input('BMI value')
    DiabeticPedigreeFunction = slt.text_input('Diabetic Pedigree value')
    Age = slt.text_input('Age value')
    
    #code for prediction.
    diagnosis = '';
    
    if slt.button('Diabetic test result'):
        diagnosis = diabetic_prediction([preganancies, BloodPressure, SkinThickness,Insulin,
                                        BMI, Age, DiabeticPedigreeFunction,glucose])

    slt.success(diagnosis)

if __name__ == '__main__':
    main()

