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

def create_BERT_dataset(source):

    input_ids = []
    attention_masks = []
    labels = []

    for index, row in source.iterrows():

        dic = tokenizer.encode_plus(row['Sentence'],                       # Sentence to encode.
                                    add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
                                    max_length = 500,           # Pad & truncate all sentences.
                                    padding = 'max_length',
                                    truncation = True,
                                    return_attention_mask = True,      # Construct attn. masks.
                                    return_tensors = 'pt',             # Return pytorch tensors.
                   )
        input_ids.append(dic['input_ids'])
        attention_masks.append(dic['attention_mask'])
        labels.append(row['multi_labels'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels, dtype=torch.float)

    dataset = TensorDataset(input_ids, attention_masks, labels)

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
            logits =  torch.nn.functional.sigmoid(logits)
            results += logits.cpu().detach().tolist()
    return results
    
    
    
if __name__ == "__main__":
    
    if not os.path.exists("./fine_tuned_baselines"): 
        print("Creating folder to store models...")
        os.mkdir("./fine_tuned_baselines")
    
    # Get device type
    device = 'cuda' if cuda.is_available() else 'cpu'
    print("Device:", device)

    print("Getting data...")
    df_train = pd.read_pickle('./data/depression/training.pkl')
    df_valid = pd.read_pickle('./data/depression/validation.pkl')
    df_test = pd.read_pickle('./data/depression/testing.pkl')
    
    print("Upsampling training data...")
    all_labels = np.array(df_train["multi_labels"].to_list())
    counts_pos = np.sum(all_labels, axis=0)
    counts_pos = np.sum(all_labels, axis=0)
    ratios = [50 / e for e in counts_pos]
    
    all_labels_false = np.array([[1 if e == 0 else 0 for e in l] for l in df_train["multi_labels"].to_list()])
    dfs_list = list()

    for i in range(21):
        sub = df_train[df_train["multi_labels"].apply(lambda x: x[i] == 1)]
        if ratios[i] < 1:
            dfs_list.append(sub)
        else:
            wholes = int(ratios[i])
            frac = int(ratios[i]) % 1
            for iter in range(wholes):
                dfs_list.append(sub)
        dfs_list.append(sub.sample(frac=frac))

    df_train_upsampled = pd.concat(dfs_list)
    df_train = df_train_upsampled
    
    print("Loading the BERT tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                                                          problem_type="multi_label_classification",
                                                          num_labels = 21, # The number of output labels--2 for binary classification.
                                                          output_attentions = False, # Whether the model returns attentions weights.
                                                          output_hidden_states = False, # Whether the model returns all hidden-states.
                                                          return_dict=False
    )
    model.to(device)
    
    print("Create training and validation and iterators...")
    training_set = create_BERT_dataset(df_train)
    validation_set = create_BERT_dataset(df_valid)
    train_iterator = DataLoader(training_set, batch_size = 16, shuffle = True)
    valid_iterator = DataLoader(validation_set, batch_size = 16, shuffle = True)
    
    if not os.path.isfile("./fine_tuned_baselines/d_model_w_upsampling_best.pt"):
        print("Model doesn't exist in cache, training now...")
        optimizer = AdamW(model.parameters(), lr=2e-5)
        train_and_evaluate(model, optimizer, train_iterator, valid_iterator, PATH, model_name="d_model_w_upsampling")
    
    print("Loading fine-tuned model...")
    checkpoint = torch.load("./fine_tuned_baselines/d_model_w_upsampling_best.pt", map_location=torch.device(device))
    print("Epoch:", checkpoint['epoch'])
    model.load_state_dict(checkpoint['model_state'])
          
    print("Creating test dataset and iterator...")
    testing_set = create_BERT_dataset(df_test)
    test_iterator = DataLoader(testing_set, batch_size = 16, shuffle = False)
    
    print("Getting predictions on test set...")
    df_test['BERT_proba'] = predict(model, test_iterator)
    df_test['BERT'] = df_test['BERT_proba'].apply(lambda x: [round(e) for e in x])
    df_test.to_pickle('./data/depression/testing.pkl')
          
    print("DONE!")