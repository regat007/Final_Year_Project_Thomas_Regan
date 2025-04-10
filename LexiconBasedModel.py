''' Lexicon based model '''
import csv
import random
import string
import nltk
import sklearn

from sklearn.metrics import classification_report

# imports for text clean up
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# load and process lexicon
def processLexicon(file):
    lexicon_dict = {}
    # open lexicon
    with open(file, encoding = 'utf-8') as text_file:
        # extract the word and classification from the lexicon
        for line in text_file:
            parts = line.split()
            word, classification = None,None
            # word and classification extraction
            for part in parts:
                if part.startswith("word1="):
                    word = part.split("=")[1]
                elif part.startswith("type="):
                    classification = part.split("=")[1]
            # add word as key and classifcation a value
            lexicon_dict[word] = classification

    return lexicon_dict

# Read in file and return sentence (sentence_id)
def readFile(sentence_id,file_name):
    # open the training data csv into csvreader
    with open(file_name, encoding = 'utf-8') as csvfile:
        csvreader = csv.reader(csvfile)

        # get rows into list object
        rows = list(csvreader)
        # remove irrelevant solved conflict column
        rows = [sublist[:-1] for sublist in rows]

        # change labels from OBJ and SUBJ to 0 and 1 respectivily
        for row in rows:
            if row[1] == 'OBJ':
                row[1] = 0
            else:
                row[1] = 1

        return rows

def dataCleaning(data):
    # clean and tokenize each sentence in the data
    for training_pair in data:
        # tokenize the sentence and remove stopwords and punctuation.
        training_pair[0] = word_tokenize(training_pair[0].lower())
        training_pair[0] = [t for t in training_pair[0] if t not in stopwords.words('english') and t not in string.punctuation]
    # retrurn the clean data
    return (data)  


# Iterate through, find word in lexicon.. +2 if strongsubj etc 
def sentenceClassifier(sentence, threshold, lexicon):
    subj_score = 0
    
    for word in sentence:
        # classify each word and update the subjectivty score
        classification = findWordsClass(word,lexicon)
        if classification == 'weaksubj':
            subj_score += 1
        elif classification == 'strongsubj':
            subj_score += 2 

    # classify depending on how many subjective words normalised by sentence length
    if (len(sentence) == 0):
        return (1, 0)  
    # subjective
    if subj_score/len(sentence) >= threshold:
        return (1, subj_score/len(sentence))
    # objective
    else:
        return (0, subj_score/len(sentence))
  
def findWordsClass(target_word,lexicon):
    # look up word in dict and retrun value (classification)
    return lexicon.get(target_word)

def wholeDataClassifier(data,lexicon):
    predictions = []
    true_values = []
    # temp: working out average scores
    subj_scores = []
    obj_scores = []
    # for each training pair make a prediciton
    for row in data:
        pred, score = sentenceClassifier(row[0],0.3,lexicon)
        predictions.append(pred)
        true_values.append(row[1])
        # temp: working out average scores
        if (row[1] == 0):
            obj_scores.append(score)
        else:
            subj_scores.append(score)

    print("Subj average = " + str(sum(subj_scores)/len(subj_scores)))
    print("Obj average = " + str(sum(obj_scores)/len(obj_scores)))

    return predictions,true_values

def metrics(preds,vals):

    report = classification_report(vals,preds)

    return report

def main():
    # pre process the lexicon and turn it into a dictionary
    lexicon = processLexicon("lexicon.txt")
    print("- Lexicon Processed -")
    # read data --> 
    data = readFile(0.3,"training_data.csv")
    print("- Data Read -")
    # clean data -->
    clean_data = dataCleaning(data)
    print("- Data Cleaned -")
    # classify -->
    predictions, true_values = wholeDataClassifier(clean_data,lexicon)
    print("- Predictions made -")
    # get metrics -->
    report = metrics(predictions,true_values)
    print(report) 


main()

