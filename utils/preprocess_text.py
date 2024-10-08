import nltk
#nltk.download('stopwords')
#nltk.download('punkt_tab')
#nltk.download('wordnet')
from nltk.corpus import stopwords
import re
#import string

#from string import punctuation

# mystem = Mystem()
# russian_stopwords = stopwords.words("russian")
#
# def preprocess_text_old(text):
#     global xczc
#     tokens = mystem.lemmatize(text.lower())
#     tokens = [token for token in tokens if token not in russian_stopwords #\
#               #and token != " " \
#               #and token.strip() not in punctuation
#               ]
#
#     text = " ".join(tokens)
#     xczc=xczc+1
#     print("pp",xczc)
#     return text

def preprocess_dig_spec(text):
    text=text.lower()
    text=re.sub(r'\d+',' ',text)
    text=re.sub(r'[^\w\s]+',' ',text)
    return nltk.word_tokenize(text)

def remove_stopwords(language,tokens):
    stop_words=set(stopwords.words(language))
    return [word for word in tokens if word not in stop_words]

def perform_lemmatization(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    tokens=preprocess_dig_spec(text)
    tokens=remove_stopwords('russian',tokens)
    tokens=remove_stopwords('english',tokens)
    tokens=perform_lemmatization(tokens)
    return ' '.join(tokens)