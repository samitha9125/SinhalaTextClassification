# Sinhala Text Classification

Text classification is currently a popular machine learning application and this repository contains details of results and datasets.


 ## Folder Structure
 
    Master -- |
              |-Code 
              |-Testing
              |-Training
              |-Validation
           
Code folder contains Code.py file which shows the implementation of this classification with data pre processing. Also it contains this trained n-grams based model in HDF5 format. 

Testing,Training,Validation Folders contain test,train and validation data respectively in categories. As an example, this Sinhala text classification is a binary classification (Classified based on Local news and International News). So these folders contain Two subfolders to represent those categories and each category contains a text.txt file (UTF-8 Encoded) which contains all news.

E.g:

    Testing -- |
               |-International - text.txt
               |-Local - text.txt
               
  ## Details 

This model has been trained using n-grams and tfidf vectorization method. It had 76% training accuracy with 70% testing accuracy. Also since it is not possible to train with sinhala fonts, **phonetic representation** used to represent sinhala words.
Under pre-processing, this algorithm only removes URLs from news.

  ## Requirements
  
      Python 3.x
      Tensorflow 1.8
      Keras 2.1.x
      sklearn
      numpy
  
  ## Run 
  
  To run this python code, from inside the Code folder run below code using terminal
 
     ./Code.py
     
  ## Re-train     
 
 You have to use another set of news under preferred number of categories and retrain the model. Also have to adjust some parameters in `Code.py`.

  ## Prediction

Here add your prediction text.txt file to `Predict` folder. Please remember to rename the file as text.txt (If you don't want to edit Code.py) and keep it as UTF-8 encoded.

For a test run of prediction, just run  `./Code.py` as said in **Run** subsection.

It is a must to have saved HDF5 file under Code folder. If not, program will automatically start to train before predict.

Note that, your prediction may not be very accurate since this model only achieved 70% of accuracy and trained only upto some vocabulary. To get better results, retrain the model with paragraphs of news having your preferred vocabulary.
