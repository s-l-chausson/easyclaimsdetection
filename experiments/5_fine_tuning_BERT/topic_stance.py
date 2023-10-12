import os
os.chdir('../..')
print(os.getcwd())

import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from torch import cuda

# Helper functions

def create_BERT_topic_dataset(source, label2idx):
    input_ids = []
    attention_masks = []
    labels = []
    for index, row in source.iterrows():
        dic = tokenizer.encode_plus(row['Tweet'],                       # Sentence to encode.
                                    add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
                                    max_length = 500,           # Pad & truncate all sentences.
                                    padding = 'max_length',
                                    truncation = True,
                                    return_attention_mask = True,      # Construct attn. masks.
                                    return_tensors = 'pt',             # Return pytorch tensors.
                   )
        input_ids.append(dic['input_ids'])
        attention_masks.append(dic['attention_mask'])
        labels.append(label2idx[row['topic_annot']])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset


def create_BERT_stance_dataset(source, labels_dict):
    input_ids = []
    attention_masks = []
    labels = []
    for index, row in source.iterrows():
        dic = tokenizer.encode_plus(row['Tweet'],                       # Sentence to encode.
                                    add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
                                    max_length = 500,           # Pad & truncate all sentences.
                                    padding = 'max_length',
                                    truncation = True,
                                    return_attention_mask = True,      # Construct attn. masks.
                                    return_tensors = 'pt',             # Return pytorch tensors.
                   )
        input_ids.append(dic['input_ids'])
        attention_masks.append(dic['attention_mask'])
        labels.append(labels_dict[row['topic_stance_annot']])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

def create_BERT_testing_dataset(source):
    input_ids = []
    attention_masks = []
    for index, row in source.iterrows():
        dic = tokenizer.encode_plus(row['Tweet'],                       # Sentence to encode.
                                    add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
                                    max_length = 500,           # Pad & truncate all sentences.
                                    padding = 'max_length',
                                    truncation = True,
                                    return_attention_mask = True,      # Construct attn. masks.
                                    return_tensors = 'pt',             # Return pytorch tensors.
                   )
        input_ids.append(dic['input_ids'])
        attention_masks.append(dic['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    dataset = TensorDataset(input_ids, attention_masks)
    return dataset


def epoch_time(start_time, end_time):
    '''Track training time. '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    
def train(model, iterator, optimizer):
    '''Train the model with specified data, optimizer, and loss function. '''
    epoch_loss = 0
    model.train()
    for batch in iterator:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        # Reset the gradient to not use them in multiple passes
        optimizer.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        logits = outputs[1]
        # Backprop
        loss.backward()
        # Optimize the weights
        optimizer.step()
        # Record loss
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def evaluate(model, iterator):
    '''Evaluate model performance. '''
    epoch_loss = 0
    # Turm off dropout while evaluating
    model.eval()
    # No need to backprop in eval
    with torch.no_grad():
        for batch in iterator:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def train_and_evaluate(model, optimizer, train_iterator, valid_iterator, path, model_name = 'model'):
    best_valid_loss = float('inf')
    best_epoch = 0
    epoch = 0
    
    while True:
        # Calculate training time
        start_time = time.time()

        # Get epoch losses and accuracies
        train_loss = train(model, train_iterator, optimizer)
        valid_loss = evaluate(model, valid_iterator)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:2} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

            #Save every epoch
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'valid_loss': valid_loss}, path + "/" + model_name + "_best.pt")
            epoch += 1
        else:
            break
        

def predict(model, iterator):
    '''Predict using model. '''
    results = list()
    # Turm off dropout while evaluating
    model.eval()
    # No need to backprop in eval
    with torch.no_grad():
        for batch in tqdm(iterator):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            _, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
            # Use softmax because MULTI-CLASS CLASSIFICATION
            logits = torch.nn.functional.softmax(logits, dim=1)
            results += logits.cpu().detach().tolist()
    return results


def get_claim(row):
    if row['Stance'] == 'AGAINST':
        return topic2index[row['Target']] + 'A'
    elif row['Stance'] == 'FAVOR':
        return topic2index[row['Target']] + 'F'
    else:
        return topic2index[row['Target']] + 'N'
    
    
    
if __name__ == "__main__":
    
    if not os.path.exists("./fine_tuned_baselines"): 
        print("Creating folder to store models...")
        os.mkdir("./fine_tuned_baselines")
    
    # Get device type
    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Device:", device)

    print("Getting data...")
    df_train = pd.read_pickle('./data/topic_stance/training.pkl')
    df_valid = pd.read_pickle('./data/topic_stance/validation.pkl')
    df_test = pd.read_pickle('./data/topic_stance/testing.pkl')
    
    print("Create label field...")
    topic2index = {
        'Atheism': '1',
        'Climate Change is a Real Concern': '2',
        'Feminist Movement': '3',
        'Hillary Clinton': '4',
        'Legalization of Abortion': '5',
    }
    df_train['topic_stance_annot'] = df_train.apply(get_claim, axis=1)
    df_valid['topic_stance_annot'] = df_valid.apply(get_claim, axis=1)
    df_test['topic_stance_annot'] = df_test.apply(get_claim, axis=1)
    
    print("STEP 1: Train TOPIC model...")
    
    label2idx = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
    }
    
    idx2label = {k: label2idx[k] for k in label2idx}
    
    print("Loading the BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                                                          num_labels = 5, # The number of output labels--2 for binary classification.
                                                          output_attentions = False, # Whether the model returns attentions weights.
                                                          output_hidden_states = False, # Whether the model returns all hidden-states.
                                                          return_dict=False
    )
    model.to(device)
    
    print("Create training and validation and iterators...")
    training_set = create_BERT_topic_dataset(df_train, label2idx)
    validation_set = create_BERT_topic_dataset(df_valid, label2idx)
    train_iterator = DataLoader(training_set, batch_size = 16, shuffle = True)
    valid_iterator = DataLoader(validation_set, batch_size = 16, shuffle = True)
    
    if not os.path.isfile("./fine_tuned_baselines/topic_model_best.pt"):
        print("Model doesn't exist in cache, training now...")
        optimizer = AdamW(model.parameters(), lr=2e-5)
        train_and_evaluate(model, optimizer, train_iterator, valid_iterator, PATH, model_name="topic_model")
    
    print("Loading fine-tuned model...")
    checkpoint = torch.load("./fine_tuned_baselines/topic_model_best.pt", map_location=torch.device(device))
    print("Epoch:", checkpoint['epoch'])
    model.load_state_dict(checkpoint['model_state'])
          
    print("Creating test dataset and iterator...")
    testing_set = create_BERT_topic_dataset(df_test, label2idx)
    test_iterator = DataLoader(testing_set, batch_size = 16, shuffle = False)
    
    print("Getting predictions on test set...")
    df_test['BERT_topic_proba'] = predict(model, test_iterator)
    df_test['BERT_topic'] = df_test['BERT_topic_proba'].apply(lambda x: idx2label[x.index(max(x))])
    df_test.to_pickle('./data/climate_change/testing.pkl')
     
    print("STEP 2: Train STANCE model...")
    
    label2idx = {
        str(topic) + "A": 0,
        str(topic) + "F": 1,
        str(topic) + "N": 2,
        }
    idx2label = {k: label2idx[k] for k in label2idx}
    
    results_test = dict()
    
    for topic in [1, 2, 3, 4, 5]:
        print("...for topic", topic)
        train_sub = df_train[df_train["topic_annot"] == topic]
        valid_sub = df_valid[df_valid["topic_annot"] == topic]
        test_sub = df_test[df_test["topic_annot"] == topic]
        
        if "topic_" + str(topic) + "_stance_model_best.pt" in os.listdir(PATH):
            print("\tAlready done")

        print('\tLoading BERT tokenizer...')
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                                                            num_labels = 3, # The number of output labels--2 for binary classification.
                                                            output_attentions = False, # Whether the model returns attentions weights.
                                                            output_hidden_states = False, # Whether the model returns all hidden-states.
                                                            return_dict=False
        )
        model.to(device)
        
        print("\tCreate datasets and iterators...")
        training_set = create_BERT_stance_dataset(train_sub, label2idx)
        validation_set = create_BERT_stance_dataset(valid_sub, label2idx)
        train_iterator = DataLoader(training_set, batch_size = 16, shuffle = True)
        valid_iterator = DataLoader(validation_set, batch_size = 16, shuffle = True)
        
        print("\tTrain the model...")
        optimizer = AdamW(model.parameters(), lr=2e-5)
        train_and_evaluate(model, optimizer, train_iterator, valid_iterator, PATH, epochs=3, model_name= "topic_" + str(topic) + "_stance_model")
        
        print("Get predictions for that topic...")
        sub_test = df_test[df_test["BERT_topic"] == topic]
        
        testing_set = create_BERT_testing_dataset(sub_test)
        test_iterator = DataLoader(testing_set, batch_size = 16, shuffle = False)

        checkpoint = torch.load(os.path.join(PATH, "topic_" + str(topic) + "_stance_model_best.pt"))
        print("Epoch:", checkpoint['epoch'])
        model.load_state_dict(checkpoint['model_state'])

        sub_test['BERT_topic_stance_proba'] = predict(model, test_iterator)
        sub_test['BERT_topic_stance'] = sub_test['BERT_topic_stance_pred_proba'].apply(lambda x: idx2label[x.index(max(x))])

        for i,row in sub_test.iterrows():
            results_test[row["Tweet"]] = row["BERT_stance"]
        
        del model
    
    df_test["BERT_topic_stance"] = df_test["Tweet"].apply(lambda x: results_test[x])
    
    print("Saving results...")
    df_test.to_pickle('./data/topic_stance/testing.pkl')
        
    print("DONE!")