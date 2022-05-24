import math
import numpy as np
import pandas as pd

"""# Model using XLNet"""


#Installing libraries 
import torch
import transformers
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report #, accuracy
from textwrap import wrap
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
import torch.nn.functional as tFn


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import pad_sequences

from tqdm import tqdm, trange
import io

from sklearn.metrics import accuracy_score

# Ignore info logging.
#from transformers import logging
#logging.set_verbosity_warning()

# reference from http://mccormickml.com/2019/09/19/XLNet-fine-tuning/

import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

# XLNet requires input sentences to be of the same length
# this is managed via use of truncate and pad

# Set the maximum sequence length. 
# title -- using 128 to account for longer titles (up to ~15 words)
# article -- using 384 to take into account first ~50 words in an article 
text_source = 'title'

MAX_LEN = 128

# Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top. 

model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)
model.cuda()

model = torch.load('models/ner-healthtechother-articles-21.pkl')
#model = torch.load('models/blah2a.pkl')
#model = model.load_state_dict(torch.load('models/blah2a.pkl'))

# Create sentence and label lists
# using selection by admin: title or article
if text_source == 'title':
    sentences = ['People with pancreatic ductal adenocarcinoma (PDAC) are six times more likely to have had gallstones the year before their diagnosis than people without this type of cancer In a new study, researchers say their findings indicate that gallstones could be a warning sign for the aggressive and often deadly form of cancer. The study will be presented this weekend at the Digestive Disease Week 2022 meeting in San Diego. The research hasn’t been peer-reviewed or published yet. PDAC is the most common form of pancreatic cancer, accounting for more than 90 percent of cases. It often is fatal because it tends to be diagnosed in later stages.']
    sentences = ['This week, we will have a high-level discussion of privacy issues. You should review the General Overview of the General Data Protection Regulation from the EU (GDPR). Technically, this regulation applies only to European citizens. However since many companies have an international customer base, the decision is often made to adhere to this regulation across all customers. Additionally, the California Consumer Privacy Act is very similar to GDPR, and many anticipate similar US-based regulations in the coming years.']
    sentences = ['Man With No Sleep, Hydration, Or Caffeine Blindsided By Inexplicable Migraine Again']

# We need to add special tokens at the beginning and end of each sentence for XLNet to work properly
sentences = [sentence + " [SEP] [CLS]" for sentence in sentences]
labels = [ 1 ]

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 

prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(labels)
  
batch_size = 1  


prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Add batch to CPU

  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)

print(logits)

# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
print(flat_predictions)

flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
print(flat_predictions)
