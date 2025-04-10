import csv
import string
import sklearn
import pandas as pd
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
# hypParams()
from torch.optim import AdamW
from transformers import get_scheduler
# training
from tqdm import tqdm
# metrics 
from sklearn.metrics import accuracy_score, classification_report




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

# cleans data (input of forms of 2D list [[,],..,[,]]) and turns into data frames
def tokenizeData(data):
    # convert data to a pands data frame
    data_frame = pd.DataFrame(data,columns=['sentence','label'])

    # load bert based tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #tokenize sentecnes
    tokenized_data = tokenizer(
        list(data_frame['sentence']),
        padding = True,
        truncation = True,
        max_length = 128,
        return_tensors = "pt"
    )
    # get labels
    labels = torch.tensor(data_frame['label'].tolist())

    return tokenized_data,labels

def splitData(data,labels):
    # define the split
    split = 0.8
    training_indices, testing_indices = train_test_split(range(len(labels)),train_size = split, stratify=labels)

    train_data = {key: tensor[training_indices] for key, tensor in data.items()}
    test_data = {key: tensor[testing_indices] for key, tensor in data.items()}
    train_labels = labels[training_indices]
    test_labels = labels[testing_indices]

    return train_data, train_labels, test_data, test_labels

def createLoaders(train_data, train_labels,test_data, test_labels):
    
    # Create datasets
    train_dataset = TensorDataset(train_data['input_ids'], train_data['attention_mask'], train_labels)
    test_dataset = TensorDataset(test_data['input_ids'], test_data['attention_mask'], test_labels)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    return train_loader, test_loader

def loadModel(model_choice):

    # select model
    model = AutoModelForSequenceClassification.from_pretrained(model_choice, num_labels=2)

    return model 

def hypParams(model,train_loader, learning_rate):

    # optimizer based on AdamW Algorithm
    optimizer = AdamW(model.parameters(), lr = learning_rate)

    # create lr scheduelr 
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=(len(train_loader)*3))

    return optimizer, lr_scheduler

def training(model,train_loader, optimizer, lr_scheduler, device):
    epochs = 3
    model.train()
    for epoch in range(epochs):
        # wrap iretable in tqdm to create progress bar
        progress_bar = tqdm(train_loader)
        # iterate through batchs in train_loader object
        for batch in progress_bar:
            # extract ids, att mask and labels 
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # forward pass...
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            # backwards pass...
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix(loss=loss.item())

    return model

def eval(model, test_loader, device):

    model.eval()

    preds, true_vals = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            true_vals.extend(labels.cpu().numpy())

    return accuracy_score(true_vals,preds), classification_report(true_vals, preds)


def main():
    # read data -->
    data = readTrainingData("training_data.csv")
    print("-Data Read-")
    # tokenized data and extract labels -->
    tokenized_data,labels = tokenizeData(data)
    print("-Data Tokenized-")
    # split data (80:20 <-> training:test) -->
    train_data, train_labels,test_data, test_labels = splitData(tokenized_data,labels)
    print("-Data Split-")
    # create datasets and dataloaders --> 
    train_loader, test_loader = createLoaders(train_data, train_labels,test_data, test_labels)
    print("-Loaders created-")
    # Load pre-trained model (BERT) -->
    model = loadModel("bert-base-uncased")
    print("-Model Loaded-")
    # define hyperparameters -->
    optimizer, lr_scheduler = hypParams(model,train_loader, 5e-5)
    # train the model if not already existed 
    try:
        trained_model = model.from_pretrained("./BertModel")
        print("Pre-trained model found...")
    except:
        print("Training model...")
        trained_model = training(model,train_loader,optimizer, lr_scheduler, torch.device('cpu'))
        model.save_pretrained("./BertModel")
    # evaluate model on test dataset
    acc_score, class_report = eval(trained_model,test_loader, torch.device('cpu'))

    print("Accuracy:" + str(acc_score))
    print(class_report)

    

main()