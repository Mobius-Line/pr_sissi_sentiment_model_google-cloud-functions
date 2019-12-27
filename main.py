from gensim.models import KeyedVectors
from os import path
from nltk.tokenize import RegexpTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import en_core_web_sm
import nltk
from google.cloud import storage



# We keep model as global variable so we don't have to reload it in case of warm invocations
the_model = None
word_vectors = None
BUCKET_NAME = 'pr_sissi_sentiment_model'

nltk.download('stopwords')

def make_token(review):

  # drop points like . . . . . . . . . ., and not useful characters. 
  tokenizer = RegexpTokenizer("\w+\'?\w+|\w+") 
  
  return tokenizer.tokenize(str(review))

def remove_stopwords(review):

    #load stop words 
    stop_words = stopwords.words('english')
    exceptionStopWords = {
      'again',
      'against',
      'ain',
      'almost',
      'among',
      'amongst',
      'amount',
      'anyhow',
      'anyway',
      'aren',
      "aren't",
      'below',
      'bottom',
      'but',
      'cannot',
      'couldn',
      "couldn't",
      'didn',
      "didn't",
      'doesn',
      "doesn't",
      'don',
      "don't",
      'done',
      'down',
      'except',
      'few',
      'hadn',
      "hadn't",
      'hasn',
      "hasn't",
      'haven',
      "haven't",
      'however',
      'isn',
      "isn't",
      'least',
      'mightn',
      "mightn't",
      'move',
      'much',
      'must',
      'mustn',
      "mustn't",
      'needn',
      "needn't",
      'neither',
      'never',
      'nevertheless',
      'no',
      'nobody',
      'none',
      'noone',
      'nor',
      'not',
      'nothing',
      'should',
      "should've",
      'shouldn',
      "shouldn't",
      'too',
      'top',
      'up',
      'very'
      'wasn',
      "wasn't",
      'well',
      'weren',
      "weren't",
      'won',
      "won't",
      'wouldn',
      "wouldn't"
}

    # union and clean basic stop words
    stop_words = set(stop_words).union(STOP_WORDS)
    final_stop_words = stop_words-exceptionStopWords

    return [token for token in review if token not in final_stop_words]

def lemmatization(review):
    
    # Part-of-speech tagging. When you switch off -  disable - parser, tagger, ner it could work more faster 
    nlp = en_core_web_sm.load(disable=['parser', 'tagger', 'ner'])
    lemma_result = []
    
    for words in review:
        doc = nlp(words)
        for token in doc:
            lemma_result.append(token.lemma_)
    return lemma_result

def pipeline(review):
    review = make_token(review)
    review = remove_stopwords(review)
    return lemmatization(review)

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout,embedding_weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, text_lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)) 
        return self.fc(hidden.squeeze(0))
    
def word2idx(embedding_model,review):
    index_review = []
    for word in review:
        try:
            index_review.append(embedding_model.vocab[word].index)
        except: 
             pass
    return torch.tensor(index_review)   

def predict_sentiment(the_model, sentence, word_vec):
    tokenized = pipeline(sentence)
    indexed = word2idx(word_vec,tokenized)
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(the_model(tensor,torch.LongTensor([len(indexed)])))
    return prediction.item()    
    
  
    
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    
    print('Blob {} downloaded to {}.'.format(source_blob_name, destination_file_name))


def handler(request):
    """input {"text_input":"film worse"} """
    
    global the_model
    global word_vectors
    
    request_json = request.get_json(silent=True)
    request_args = request.args
    
    if request_json and 'text_input' in request_json:
        name = request_json['text_input']
        print(name)        
    elif request_args and 'text_input' in request_args:
        name = request_args['text_input']
        print(name)
    else:
        name = 'film worse'
    
    
    # Model load which only happens during cold starts
    if None in (the_model, word_vectors):
        
        download_blob(BUCKET_NAME, 'sissi_sentiment_12122019.pth',
                  '/tmp/sissi_sentiment_12122019.pth')
        
        download_blob(BUCKET_NAME, 'word2vec.model',
                  '/tmp/word2vec.model')
        
        word_vectors = KeyedVectors.load('/tmp/word2vec.model')
        embedding_weights = torch.Tensor(word_vectors.vectors)
        padding_value = len(word_vectors.index2word)
        
                
        INPUT_DIM = padding_value
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5
        the_model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, embedding_weights)
        the_model.load_state_dict(torch.load('/tmp/sissi_sentiment_12122019.pth', map_location='cpu'))
        

        
    predictions = predict_sentiment(the_model, name, word_vectors) 
    print(predictions)
    data = [str(predictions)]
    
    return data[0]