from typing import List, Tuple
import numpy as np
import torch
import transformers
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification
from textwrap import wrap
from torch import optim
from torch import nn
import torch.nn.functional as tFn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import pad_sequences

# Ignore info logging.
#from transformers import logging
#logging.set_verbosity_warning()

################################################################################
# Model utilities.
def load(model_path: str) -> Tuple[object, object, object, object]:
    """Load the model.

    Parameters
    ----------
    model_path : str
        Path to the model file.
    
    Returns
    -------
    object
        The loaded model.
    """
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    # Load XLNEtForSequenceClassification, the pretrained XLNet model with a single linear classification layer on top. 
    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=3)
    model.cuda()

    model = torch.load(model_path)

    # Put model in evaluation mode
    model.eval()

    return model, device, tokenizer


def classify(model: object, device: object, tokenizer: object, maxlen: int, query_string: str, ner_labels: List) -> Tuple[str, int, str]:
    # We need to add special tokens at the beginning and end of each sentence for XLNet to work properly
    query_list = [query_string]
    query_list = [sentence + " [SEP] [CLS]" for sentence in query_list]
    labels = [ 1 ]

    tokenized_texts = [tokenizer.tokenize(sent) for sent in query_list]

    # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=maxlen, dtype="long", truncating="post", padding="post")
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

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

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

    # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    flat_predictions = [item for sublist in predictions for item in sublist]
    class_num = np.argmax(flat_predictions, axis=1).flatten()[0]

    return query_string, class_num, ner_labels[class_num]
