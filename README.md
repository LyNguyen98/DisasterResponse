# Disaster Response

**1. About the project**

This project is using Machine learning to approach and process data in order to build a predictive model that is able to help user identify messages that carrying important infomation about life threatening situation. This project aims to help users quickly identify emergency messages and its context and act promptly. This is very crucial, especially in natural disaster situation, for a person to act quickly and accurately.

**2.	Motivation**

This is my second project to complete the Nanodegree course in Data science provided by Udacity.

**3.	Prerequisites**

  •	Pandas version 1.4.2

  •	Numpy version 1.21.5

  •	Matplotlib version 3.5.1
  
  •	Natural Language Processing: NLTK
  
  •	SQLalchemy
  
  •	Sklearn: Pipeline, model_selection, ensemble, feature_extraction.text, metrics

**4.	Explaination**

     app
    | - template
    | |- master.html # main page of web app
    | |- go.html # classification result page of web app
    |- run.py # Flask file that runs app
    data
    |- disaster_categories.csv # data to process
    |- disaster_messages.csv # data to process
    |- process_data.py
    |- DisasterResponse.db # database to save clean data to
    models
    |- train_classifier.py
    |- classifier.pkl # saved model
    README.md
  

**4.	Contact**

  •	Project link: https://github.com/LyNguyen98/DisasterResponse
  
**5.	Tutorial**

  •	run cmd: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv
  
  •	run cmd: python models/train_classifier.py data/DisasterResponse.db
  
  •	cd to the apps folder and run cmd: python run.py
  

**6.	Acknowledgement:**

  •	Author: Ly Nguyen
  
  •	Udacity course: udacity.com

