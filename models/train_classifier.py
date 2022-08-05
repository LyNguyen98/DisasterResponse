import sys
import pandas as pd 
import numpy as np
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
from sqlalchemy import create_engine 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def load_data():
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse',engine)
    df['related'] = df['related'].apply(lambda x: 1 if x==2 else x)
    X = df['message']
    y = df.loc[:,'related':]
    return X, y


def tokenize(text):
    #find url text and replace with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detect_url = re.findall(url_regex, text)
    for url in detect_url:
        text = text.replace(url, 'urlplaceholder')
    
    #Tokenize and lemmatizer
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    #building pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    return pipeline

def display_result(y_pred, y_test):
    print('Labels: ', np.unique(y_pred))
    #convert array to dataframe
    y_pred_df = pd.DataFrame(y_pred, columns = y_test.columns)
    for col in y_test.columns:
        print("Category column name: ", col)
        print(classification_report(y_test[col], y_pred_df[col]))


def save_model(pipeline):
    pickle.dump(pipeline, open('models/classifier.pkl', 'wb'))


def main():
        #load data
        X, y = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        #build & fit model
        model = build_model()
        model.fit(X_train, y_train)
        
        #display model result
        y_pred = model.predict(X_test)
        display_result(y_pred, y_test)

        #save model
        save_model(model)


if __name__ == '__main__':
    main()
