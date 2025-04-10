
import csv 
import string
import nltk
import sklearn
import numpy as np

# imports for text clean up
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# imports for svm model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score



# read in training data and test data
def readTrainingData(training_data_file):
    # open training data csv
    with open(training_data_file, encoding = 'utf-8') as csvfile:
        csvreader = csv.reader(csvfile)

        # create 2d list of [sentence,label,solved_conflict] tuples
        rows = list(csvreader)
        # remove solved_conflict from the list as it is irrellevant
        rows = [sublist[:-1] for sublist in rows]

        # change labels from OBJ and SUBJ to 0 and 1 respectivily
        for row in rows:
            if row[1] == 'OBJ':
                row[1] = 0
            else:
                row[1] = 1

        return (rows)

# cleans data (input of forms of 2D list [[,],..,[,]])
def dataCleaning(data):
    # clean and tokenize each sentence in the data
    for training_pair in data:
        # tokenize the sentence and remove stopwords and punctuation.
        training_pair[0] = word_tokenize(training_pair[0].lower())
        training_pair[0] = [t for t in training_pair[0] if t not in stopwords.words('english') and t not in string.punctuation]
    # retrurn the clean data
    return (data)    

def vectorize_data(data):
    
    # join tokens to make setences again and seperate from labels
    sentences = [' '.join(tokens) for tokens,_ in data]
    labels = [label for _,label in data]

    # tf-idf the sentences
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    return X,labels,sentences

# split data into training data and test data
def splitData(X,y):

    # split data 80/20 train/test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

    return X_train, X_test, y_train, y_test

# build and train an SVM then make predictions 
def supportVectorMachine(X_train,y_test,X_test,y_train):
    # initialse the svm (c = regulalisation strength)
    svm_model = SVC(kernel='linear', class_weight = 'balanced', C = 1.0, probability = True)

    # train the model
    svm_model.fit(X_train,y_train)

    # make predicitons 
    predictions = svm_model.predict(X_test)
    probabilities = svm_model.predict_proba(X_test)[:,1]

    predictions = (probabilities >= 0.45).astype(int)

    return predictions,probabilities

def performanceMetrics(predictions,probabilities,y_test):

    # produce metrics
    accuracy = accuracy_score(y_test,predictions)
    metrics_report = classification_report(y_test,predictions)
    roc_auc_metrics = roc_auc_score(y_test,probabilities)

    # display metrics
    print("Accuracy: " + str(accuracy))
    print("Metrics report :")
    print(metrics_report)
    print("ROC AUC: ")
    print(roc_auc_metrics)

    return accuracy,metrics_report,roc_auc_metrics



def main():
    # read data -->
    data = (readTrainingData("training_data.csv"))
    # clean date --> 
    clean_data = dataCleaning(data)
    # vectorize data -->
    vectorized_data,labels,sentences = vectorize_data(clean_data)
    # split data --> 
    X_train, X_test, y_train, y_test = splitData(vectorized_data,labels)
    # Build, train and predict -->
    preds,probs = supportVectorMachine(X_train,y_test,X_test,y_train)
    # Calculate performance metrics --> 
    accuracy, metrics_report, roc_auc_metric = performanceMetrics(preds,probs,y_test)
    

main()