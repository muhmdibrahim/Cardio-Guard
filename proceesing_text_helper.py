
import re
import nltk
import string, numpy as np

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORD = set(stopwords.words('english'))

import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")


from nltk import WordNetLemmatizer
from nltk.stem import SnowballStemmer

snowball_stemmer = SnowballStemmer('english')
lemma = WordNetLemmatizer()


#Uniform whitespaces
def uniform_whitespaces(text):
    if type(text) == str :
        text_clean = re.sub(' +', ' ', text )
        return text_clean
    else :
        return np.nan 
    


#Remove punctuation
def remove_punctuation(text):
    if type(text) == str :
        text_clean = "".join(c for c in text if c not in string.punctuation)
        return text_clean
    else :
        return np.nan
    
#Remove Number
def remove_number(text):
    if type(text) == str :
        text_clean = re.sub(r'\d+', '', text )
        return text_clean
    else :
        return np.nan
    
#Remove stopwords
def remove_stopwords(text):
    STOPWORD.update(('and','I','A','http','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','mailto','regards','ayanna','like','email'))
    if type(text) == str :
        text_clean = [w.lower() for w in text.split() if w.lower() not in STOPWORD]
        return " ".join(text_clean)
    else :
        return np.nan
    

#Remove Links
def remove_links(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'bit.ly/\S+', '', text)
    text = text.strip('[link]')
    return text


def remove_emails(text):
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
    cleaned_text = email_pattern.sub('', text)
    return cleaned_text


# Clean Text
def clean_text(text):
    text_clean = text.replace('\xa0', '')

    text_clean = word_tokenize(text_clean)

    text_clean = [word.lower() for word in text_clean]

    text_clean = [re.sub('[^A-Za-z]', '', word) for word in text_clean]

    text_clean = [w for w in text_clean if len(w) > 1]

    text_clean = [w for w in text_clean if ' ' not in w]

    return " ".join(text_clean)


#Stemming Words
def Stemming(text):
    stemming = []
    tokens = nltk.word_tokenize(text)
    stem_word = [ snowball_stemmer.stem(word) for word in tokens]
    stemming = " ".join(stem_word)
    return stemming 


#Lemmatize Text
def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_tokens = []
    for token in doc:
        if token.pos_ in ['ADJ', 'VERB', 'NOUN', 'ADV']:
            lemmatized_token = token.lemma_
        else:
            lemmatized_token = token.text
        lemmatized_tokens.append(lemmatized_token)
    return " ".join(lemmatized_tokens) 




#Convert To Lower Case
def lower(text):
    return text.lower()





#Apply All preprocessing
def preprocessing(text):
    text = text.apply(lower)
    text = text.apply(clean_text)
    text = text.apply(remove_emails)
    text = text.apply(remove_links)
    text = text.apply(remove_number)
    text = text.apply(remove_punctuation)
    text = text.apply(uniform_whitespaces)
    text = text.apply(remove_stopwords)
    text = text.apply(lemmatize_text)

    return text 
