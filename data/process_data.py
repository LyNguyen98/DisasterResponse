import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'inner', on = 'id')
    return df


def clean_data(df):
    categories = df['categories'].str.split(';', expand = True)
    
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str[0]
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype('int')
     
    #Drop category columns and concatnate with category table
    df.drop('categories', axis = 1, inplace = True)
    df = df.merge(categories, left_index = True, right_index = True)
    
    #drop duplicates
    df.drop_duplicates(keep = 'first', inplace = True)
    
    return df
    


def save_data(df):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 3:

        messages_filepath, categories_filepath= sys.argv[1:]

        df = load_data(messages_filepath, categories_filepath)

        df = clean_data(df)
        
        save_data(df)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()

