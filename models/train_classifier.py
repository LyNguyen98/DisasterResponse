import sys
import pandas as pd 
import numpy as np
import nltk
import re
import pickle
from sklearn.model_selection import GridSearchCV
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


def load_data(database_filepath):
    '''
    Load data from saved database
    database_filepath   path of saved database
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    y = df.loc[:,'related':]
    return X, y


def tokenize(text):
    '''
    Tokenize and Lemmatize text
    '''
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
    '''
    Building pipeline
    
    And return a pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    parameters = {
        'clf__n_estimators': [30, 60, 90],
        'clf__max_depth' : [4,5,6]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters) 
    
    return cv

def display_result(y_pred, y_test):
    '''
    Show model labels
    
    and accuracy
    '''
    print('Labels: ', np.unique(y_pred))
    #convert array to dataframe
    y_pred_df = pd.DataFrame(y_pred, columns = y_test.columns)
    for col in y_test.columns:
        print("Category column name: ", col)
        print(classification_report(y_test[col], y_pred_df[col]))


def save_model(pipeline, model_name):
    '''
    Save model
    '''
    pickle.dump(pipeline, open(model_name, 'wb'))


def main():
    #load data
    if len(sys.argv) == 3:

        database_filepath, model_name= sys.argv[1:]
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        #build & fit model
        model = build_model()
        model.fit(X_train, y_train)
        
        #display model result
        y_pred = model.predict(X_test)
        display_result(y_pred, y_test)

        #save model
        save_model(model,model_name)
    else:
        print('Please provide enough filepath')


if __name__ == '__main__':
    main()
