#!/usr/bin/env python
# coding: utf-8

# ## **Machine Learning**

# ### **Train-Test Split**
# 
# We split the data into train and test with the proportion 75:25

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score ,classification_report,confusion_matrix

x = df['tweet_clean']
y,class_names = pd.factorize(df['cyberbullying_type'],sort=True)

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=5)

x_train = [str(x) for x in X_train]
x_test  = [str(x) for x in X_test]


# ### **What is Bi-LSTM?**
# LSTM (Long Short-Term Memory) is a renowned type of recurrent neural network (RNN) used for processing sequential data and capturing long-term dependencies. We employ Bidirectional-LSTM (Bi-LSTM) to extend this capability by processing input sequences in both forward and backward directions, effectively capturing contextual information.
# 
# ### **Architecture**
# 
# The model architecture incorporates:
# 
# - An input embedding layer initialized with pre-trained word embeddings from Word2Vec, capturing the semantic meanings of tweets;
# 
# - A core LSTM layer, processing input sequences bidirectionally to capture nuanced relationships between words;
# 
# - An attention mechanism, focusing on important parts of the input sequence and enhancing the model's classification performance.
# 
# ### **Training Parameters**
# 
# During training, the model parameters are optimized using the AdamW optimizer and minimize the negative log-likelihood loss function (NLLLoss). We define hyperparameters such as:
# 
# *Number of classes:* 6
# 
# *Hidden dimensions:* 100
# 
# *Number of LSTM layers:* 1
# 
# *Dropout rate*
# 
# *Learning rate*
# 
# *Number of epochs:* 10
# 
# Epochs refer to the number of times the entire dataset is passed forward and backward through the neural network during training. We implement early stopping based on validation accuracy to prevent overfitting.
# 
# ### **Model Evaluation**
# 
# The best-performing model, based on validation accuracy, is selected for evaluation. After training, we evaluate the model on the test data and compute the classification report. The model achieves an accuracy of 0.81, demonstrating the effectiveness of our LSTM-based approach in cyberbullying tweet classification.

# In[24]:


from gensim.models import Word2Vec

# Train Word2Vec model
Word2vec_train_data = list(map(lambda x: x.split(), x_train))
EMBEDDING_DIM = 200
word2vec_model = Word2Vec(Word2vec_train_data, vector_size=EMBEDDING_DIM)
print(f"Vocabulary size: {len(vocabulary) + 1}")


# In[25]:


# Define a function to map sentiment labels to numerical values
df['sentiment'] = df['cyberbullying_type'].replace({
    'religion': 5,
    'age': 0,
    'ethnicity': 1,
    'gender': 2,
    'not_cyberbullying': 3,
    'other_cyberbullying': 4
})

# Define embedding matrix
VOCAB_SIZE = len(vocabulary) + 1
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

# Fill the embedding matrix with pre-trained values from word2vec
for word, token in vocabulary:
    if word in word2vec_model.wv.key_to_index:
        embedding_vector = word2vec_model.wv[word]
        embedding_matrix[token] = embedding_vector
print("Embedding Matrix Shape:", embedding_matrix.shape)


# ### **Setting up Model Parameters and Architecture for Training**

# In[28]:


from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import RandomOverSampler
import torch

X = tokenized_column
y = df['sentiment'].astype(np.int64).to_numpy()

# Split data into train, test, and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=5)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

# Apply Random Over Sampling
ros = RandomOverSampler()
X_train_os, y_train_os = ros.fit_resample(X_train, y_train)

# Convert data into PyTorch DataLoader
BATCH_SIZE = 32
train_data = TensorDataset(torch.from_numpy(X_train_os), torch.from_numpy(y_train_os))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)
test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)


# In[29]:


import torch.nn as nn

# Define the Attention module
class Attention(nn.Module):
    def __init__(self, hidden_dim, is_bidirectional):
        super(Attention, self).__init__()
        self.is_bidirectional = is_bidirectional
        self.attn = nn.Linear(hidden_dim * (4 if is_bidirectional else 2), hidden_dim * (2 if is_bidirectional else 1))
        self.v = nn.Linear(hidden_dim * (2 if is_bidirectional else 1), 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        if self.is_bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            hidden = hidden[-1]
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        attn_weights = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attn_weights = self.v(attn_weights).squeeze(2)
        return nn.functional.softmax(attn_weights, dim=1)

# Define the LSTM Sentiment Classifier
class LSTM_Sentiment_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, lstm_layers, dropout, is_bidirectional):
        super(LSTM_Sentiment_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = lstm_layers
        self.is_bidirectional = is_bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=is_bidirectional)
        self.attention = Attention(hidden_dim, is_bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if is_bidirectional else 1), num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.lstm(embedded, hidden)
        attn_weights = self.attention(hidden[0], out)
        context = attn_weights.unsqueeze(1).bmm(out).squeeze(1)
        out = self.softmax(self.fc(context))
        return out, hidden

    def init_hidden(self, batch_size):
        factor = 2 if self.is_bidirectional else 1
        h0 = torch.zeros(self.num_layers * factor, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers * factor, batch_size, self.hidden_dim).to(DEVICE)


# In[30]:


NUM_CLASSES = 6
HIDDEN_DIM = 100
LSTM_LAYERS = 1
IS_BIDIRECTIONAL = False
LR = 4e-4
DROPOUT = 0.5
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LSTM_Sentiment_Classifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, LSTM_LAYERS, DROPOUT, IS_BIDIRECTIONAL)
model = model.to(DEVICE)

# Initialize the embedding layer with the previously defined embedding matrix
model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
model.embedding.weight.requires_grad = True

# Set up the criterion (loss function) and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-6)
print(model)


# In[31]:


print(DEVICE)


# ### **Model Training**

# In[32]:


# Train the model
total_step = len(train_loader)
total_step_val = len(valid_loader)

early_stopping_patience = 4
early_stopping_counter = 0
valid_acc_max = 0

for e in range(EPOCHS):
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    y_train_list, y_val_list = [], []
    correct, correct_val = 0, 0
    total, total_val = 0, 0
    running_loss, running_loss_val = 0, 0

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        h = model.init_hidden(labels.size(0))
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output, labels)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        y_pred_train = torch.argmax(output, dim=1)
        y_train_list.extend(y_pred_train.squeeze().tolist())
        correct += torch.sum(y_pred_train == labels).item()
        total += labels.size(0)
    train_loss.append(running_loss / total_step)
    train_acc.append(100 * correct / total)

    with torch.no_grad():
        model.eval()
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            val_h = model.init_hidden(labels.size(0))
            output, val_h = model(inputs, val_h)
            val_loss = criterion(output, labels)
            running_loss_val += val_loss.item()
            y_pred_val = torch.argmax(output, dim=1)
            y_val_list.extend(y_pred_val.squeeze().tolist())
            correct_val += torch.sum(y_pred_val == labels).item()
            total_val += labels.size(0)
        valid_loss.append(running_loss_val / total_step_val)
        valid_acc.append(100 * correct_val / total_val)

    if np.mean(valid_acc) >= valid_acc_max:
        torch.save(model.state_dict(), './state_dict.pt')
        print(f'Epoch {e+1}:Validation accuracy increased ({valid_acc_max:.6f} --> {np.mean(valid_acc):.6f}).  Saving model ...')
        valid_acc_max = np.mean(valid_acc)
        early_stopping_counter = 0
    else:
        print(f'Epoch {e+1}:Validation accuracy did not increase')
        early_stopping_counter += 1

    if early_stopping_counter > early_stopping_patience:
        print('Early stopped at epoch :', e+1)
        break

    print(f'\tTrain_loss : {np.mean(train_loss):.4f} Val_loss : {np.mean(valid_loss):.4f}')
    print(f'\tTrain_acc : {np.mean(train_acc):.3f}% Val_acc : {np.mean(valid_acc):.3f}%')


# ### **Model Evaluation**

# In[33]:


model.load_state_dict(torch.load('./state_dict.pt'))
sentiments = ["age", "ethnicity", "gender", "other_cyberbullying", "not bullying","religion"]

def evaluate_model(model, test_loader):
    model.eval()
    y_pred_list = []
    y_test_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            test_h = model.init_hidden(labels.size(0))
            output, val_h = model(inputs, test_h)
            y_pred_test = torch.argmax(output, dim=1)
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())
    return y_pred_list, y_test_list

y_pred_list, y_test_list = evaluate_model(model, test_loader)
print('Classification Report for Bi-LSTM :\n', classification_report(y_test_list, y_pred_list, target_names=sentiments))


# ### **Takeaways**
# 
# Advantages
# 
# - Proccesses input sequences in both forward and backward directions helps in understanding the complete context of the input sequence.
# 
# - Well-suited for capturing long-term dependencies in sequential data and can effectively model complex dependencies over extended sequences.
# 
# - Random Forest can efficiently handle large datasets with many features and instances, making it suitable for complex problems.
# 
# - The gated architecture of LSTM cells helps mitigate the vanishing gradient problem, making it more capable of learning and retaining information over long sequences.
# 
# Disadvantages
# 
# - Effectively doubles the computational cost of processing each input sequence compared to unidirectional LSTMs.
# 
# - Require more memory to store the activations and gradients for both forward and backward processing directions.
# 
# - Complex models with multiple layers and bidirectional processing, make them less interpretable compared to simpler models.
# 
# - Prone to overfitting, especially when trained on small datasets or when the model capacity is too high relative to the dataset size.

# ### **Saving the complete model checkpoint into Weights (.pt) and Artifacts (.pkl)**
# Save the model weights into `biLSTM_model_weights.pt`, and bundle the model hyperparameters, vocabulary, and embedding matrix into `biLSTM_artifacts.pkl` so it can be fully reconstructed later.

# In[37]:


import torch
import pickle

# 1. Save ONLY the model weights using PyTorch
torch.save(model.state_dict(), './biLSTM_model_weights.pt')
print("Model weights saved to ./biLSTM_model_weights.pt")
files.download('./biLSTM_model_weights.pt')
print("Model weights downloaded to local machine")

# 2. Bundle the rest of the artifacts into a dictionary
artifacts = {
    'vocabulary': vocabulary,
    'embedding_matrix': embedding_matrix,
    'model_args': {
        'vocab_size': VOCAB_SIZE,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_classes': NUM_CLASSES,
        'lstm_layers': LSTM_LAYERS,
        'dropout': DROPOUT,
        'is_bidirectional': IS_BIDIRECTIONAL
    },
    'class_names': sentiments
}

# 3. Save the artifacts dictionary using the standard pickle library
with open('./biLSTM_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print("Artifacts saved to ./biLSTM_artifacts.pkl")
files.download('./biLSTM_artifacts.pkl')
print("Artifacts downloaded to local machine")

