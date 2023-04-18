import pandas as pd
import xml.etree.ElementTree as et
import numpy as np
from gensim.test.utils import datapath
from gensim import  models
import gensim.corpora as corpora
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import * 
from nltk.corpus import stopwords
import re
import gensim
import gensim.corpora as corpora
from gensim import  models
import spacy
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('vader_lexicon')

from spacy.lang.en import English
parser  = English()

dictionary = corpora.Dictionary.load('lda_model/dictionary.gensim')
TOPIC_MAP = {0: 'menu', 1: 'service', 2: 'miscellaneous', 3: 'place'}
lda_model = models.ldamodel.LdaModel.load("lda_model/newmodel")
def get_lemma(word):
    """Lemmatize (get root word) for a given word."""
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def tokenize(text):
    """Creates tokens for LDA model. Passes over all whitespaces, adds special tokens for URLs and screen names. 
    Puts all tokens in lower case."""
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens 

def prepare_text_for_lda(text):
    """Generate list of tokens, keeping all tokens that are >2 characters, and are not in the stopword list.
    Lemmatize each token."""
    en_stop = set(nltk.corpus.stopwords.words('english'))

    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 2 and "'" not in token]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def lda_prediction(restaurant_review):
    clean_sample = prepare_text_for_lda(restaurant_review)
    print(clean_sample)
    sample_2bow = dictionary.doc2bow(clean_sample)
    print(sample_2bow)
    topics = lda_model.get_document_topics(sample_2bow)
    topic_dict = {TOPIC_MAP[x[0]]:x[-1] for x in topics}
    print(topic_dict)
    top_topic = max(topic_dict.items(), key=lambda x:x[1])
    return top_topic[0]











"""def mergeDictionary_1(dict_1, dict_2):
  dict_3 = dict_1.copy()
  for key, value in dict_2.items():
    dict_3[key] = dict_2[key]
  return dict_3

def read_data_in_pd(xml_name):
  xtree = et.parse(xml_name)
  xroot = xtree.getroot() 
  row = []
  for node in xroot:
      a = {}
      b = {}
      s_text = node.find("text").text if node is not None else None
      b["text"] = s_text
      #print(s_text)
      s_aspectCategories = node.find("aspectCategories")
      for aspectCategory in s_aspectCategories.iter('aspectCategory'):
        a[aspectCategory.attrib['category']] = aspectCategory.attrib['polarity']
        
        #a = mergeDictionary(a,aspectCategory.attrib)
      b["sentiment"] = a
      b = mergeDictionary_1(b, a)
      row.append(b) 
  output_df = pd.DataFrame(row,)
  output_df = output_df.where(pd.notnull(output_df), None)
  return output_df

def preprocess_text(text,tokenizer,lrStem,wnLemm,stopWords):
    
    file_words = tokenizer.tokenize(text)
    file_words = [word.lower() for word in file_words]      
    #file_words = [lrStem.stem(word) for  word in file_words]
    file_words = [wnLemm.lemmatize(word, 'v') for word in file_words]
    file_words = [word for word in file_words if not word in stopWords]
    #file_words = " ".join(file_words)
    return file_words

def preprocess_data_text(df):
  tokenizer = RegexpTokenizer(r'\w+')
  lrStem = LancasterStemmer()
  wnLemm = WordNetLemmatizer()
  stopWords = set(stopwords.words('english'))
  df_test_1 = df.copy()
  df_test_1["text_processed"] = df_test_1.apply(lambda row: preprocess_text(row["text_processed"],tokenizer,lrStem,wnLemm,stopWords), axis=1)
  return df_test_1

def predict_lda(input_str):
  train_df = read_data_in_pd("D:/nlp_project/raw/train.xml")
  train_df.loc[train_df["food"].isnull(), "food"] = train_df.loc[train_df["food"].isnull(), "menu"]
  train_df.loc[train_df["place"].isnull(), "place"] = train_df.loc[train_df["place"].isnull(), "ambience"]
  train_df.loc[train_df["staff"].isnull(), "staff"] = train_df.loc[train_df["staff"].isnull(), "service"]
  train_df["text_processed"]= \
  train_df["text"].map(lambda x: re.sub('[,\.!?]', '', x))
  train_df_1 = preprocess_data_text(train_df)
  data_words = train_df_1.text_processed.values.tolist()
  # Create Dictionary
  id2word = corpora.Dictionary(data_words)

  # Create Corpus
  texts = data_words

  # Term Document Frequency
  #corpus = [id2word.doc2bow(text) for text in texts]
  temp_file = datapath("lda_model/newmodel")
  lda = models.ldamodel.LdaModel.load(temp_file)
  new_text_corpus =  id2word.doc2bow(input_str.split())
  score = lda[new_text_corpus]
  score_list = {}
  #sort_list = []
  for i in score:
    a,b = i
    
    if a == 0:
      score_list["staff"] = b
    if a == 1:
      score_list["food"] = b
    if a == 2:
      score_list["place"] = b
    if a == 3:
      score_list["price"] = b
  n = 1
 

 
  L = sorted(score_list.items(),key=lambda item:item[1],reverse=True)
 
  L = L[:n]
 
 
  dictdata = {}
  for l in L:
    dictdata[l[0]] = l[1]
  for i,j in dictdata.items():
    return i"""
  
def sentiment_analysis_flair(i):
   sid = SentimentIntensityAnalyzer()
   vader_score = sid.polarity_scores(i)
   if vader_score['compound'] > 0:
    value = 'positive'
   elif vader_score['compound'] < 0:
    value = "negative"

   else:
    value = "neutral"
   return value

def tagging_function(sentence):
  nlp = spacy.load('en_core_web_sm')
  doc = nlp(sentence)
  rows = {}
  subject = []
  descriptors = []
  for token in doc:
      text,tag = token.text, token.dep_
      #print(text,tag)
      if token.dep_ == "nsubj":
        subject.append(text)
      if token.dep_ == "dobj" or token.dep_ == "pobj" or token.dep_ == "acomp" or token.dep_ == "amod" or token.dep_ == "conj":
        descriptors.append(text)
  rows["subject"] = subject
  rows["descriptors"] = descriptors
  return rows

def restaurant_review(sent):
  row = []
  sent = sent.lower()  
  sent = sent.replace("!", ".").replace("?",".").replace(";",".").replace("but",".")#.replace("and",".").replace("but",".")
  sent = sent.split('.')
  print(sent)
  for i in sent:
    if len(i) > 6: #Using length as a fliter to reduce noise or meaningless phase
      dict_1 = {}
      topic_dict = lda_prediction(i)
      sentiment_dict = sentiment_analysis_flair(i)
      sub_obj_dict = tagging_function(i)
      dict_1["phase"] = i
      dict_1["topic"] = topic_dict
      dict_1["sentiment"] = sentiment_dict
      dict_1["subject"] = sub_obj_dict["subject"]
      dict_1["descriptors"] = sub_obj_dict["descriptors"]
      row.append(dict_1)
  output_df = pd.DataFrame(row,)
  return(output_df)
