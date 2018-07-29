import re,scipy
import numpy as np
import os,io,random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,TensorBoard
from keras.models import load_model
from scipy.sparse import hstack,csr_matrix
from scipy import sparse


# Global list of categories. 
categories = ['Local','International'] # All interested categories should be listed here.

# Pre-process data. Here only have removed links from sinhala news data.
def pre_process(folderName,functionType):
    path_to_Folder_Parent_Folder = os.path.abspath(__file__ + "/../../")

    if(os.path.isfile(path_to_Folder_Parent_Folder+'/'+functionType+'/'+folderName+'/text.txt') and not (os.path.isfile(path_to_Folder_Parent_Folder+'/'+folderName+'/output.txt'))):
        inFile = open(path_to_Folder_Parent_Folder+'/'+functionType+'/'+folderName+'/text.txt','r')
        outFile = open(path_to_Folder_Parent_Folder+'/'+functionType+'/'+folderName+'/output.txt','w+')
        text = inFile.read().splitlines()
        for line in text:
            outFile.write(re.sub(r'https?:\/\/.*[\r\n]*','',line)+'\n') # Use to remove links from original news articles
        outFile.close()
        inFile.close()
    else:
        print('File not found or output file exist')
        pass

# Create corpus of words from pre-processed files
def createCorpus(functionType):
    if functionType == 'Training':
        path_to_Folder_Parent_Folder = os.path.abspath(__file__ + "/../../")
        fullCorpus = []
        labels = []
        for types in categories:
            if not os.path.isfile(path_to_Folder_Parent_Folder+'/'+functionType+'/'+types+'/output.txt'): 
                pre_process(types,functionType) # Create pre_processed output file if it does not exist.

            inFile = open(path_to_Folder_Parent_Folder+'/'+functionType+'/'+types+'/output.txt')
            text = inFile.read().splitlines()
            i=0
            while i<len(text):
                if types=='Local':
                    labels.append(1)
                else:
                    labels.append(0)
                i += 1
            fullCorpus = fullCorpus+ text 

        combined = list(zip(fullCorpus, labels))
        random.shuffle(combined)
        fullCorpus[:], labels[:] = zip(*combined)
        count = 0
        while count < len(labels):
            #print(fullCorpus[count]+ ' -!!- '+labels[count]) Use to print corpus and labels
            count += 1

        #print(len(labels)) Size of the corpus of all categories
        return fullCorpus,labels
        
    else:
        path_to_Folder_Parent_Folder = os.path.abspath(__file__ + "/../../")
        fullCorpus = []
        labels = []
        for types in categories:
            if not os.path.isfile(path_to_Folder_Parent_Folder+'/'+types+'/output.txt'): 
                pre_process(types,functionType) # Create pre_processed output file if it does not exist.
            inFile = open(path_to_Folder_Parent_Folder+'/'+functionType+'/'+types+'/output.txt')
            text = inFile.read().splitlines()
            i=0
            while i<len(text):
                if types=='Local':
                    labels.append(1)
                else:
                    labels.append(0)
                i += 1
            fullCorpus = fullCorpus+ text 

        combined = list(zip(fullCorpus, labels))
        random.shuffle(combined)
        fullCorpus[:], labels[:] = zip(*combined)
        count = 0
        while count < len(labels):
            #print(fullCorpus[count]+ ' -!!- '+labels[count]) Use to print corpus and labels
            count += 1

        #print(len(labels)) Size of the corpus of all categories
        return fullCorpus,labels

# Create bigram (Since n=2 here) and using TFIDF vectorization method transforme data to vector representation. SelectKBest removes all but the k highest scoring features.
def ngramFunc(training_vals,training_labels,test_values,val_values): # n-gram n value
    ngram = (1, 2)
    topK = 20000
    token_Mode = 'word'
    min_freq = 2
    kwargs = {
            'ngram_range': ngram,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': token_Mode,  # Split text into word tokens.
            'min_df': min_freq,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    training = vectorizer.fit_transform(training_vals)
    testing = vectorizer.transform(test_values)
    validating = vectorizer.transform(val_values)

    selector = SelectKBest(f_classif, k=min(topK, training.shape[1])) # Use to identify best features and use to train model and test model.
    selector.fit(training, training_labels)
    train = selector.transform(training).astype('float32')
    test = selector.transform(testing).astype('float32')
    validation = selector.transform(validating).astype('float32')

    return train,test, validation

def ngramFuncPrediction(predict_values):
    ngram = (1, 2)
    trainingFeatureSize = 1196
    token= 'word'
    min_freq = 1
    kwargs = {
            'ngram_range': ngram,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': token,  # Split text into word tokens.
            'min_df': min_freq,
            'max_features':trainingFeatureSize,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    predict = vectorizer.fit_transform(predict_values).astype('float32')
    firstSize = predict.shape[1]
    rep = trainingFeatureSize/firstSize-1
    i=0
    print(firstSize)
    temp = predict
    while i<rep:
        predict = hstack([predict,temp])
        i+=1
    calc = abs(predict.shape[1] - trainingFeatureSize)
    predict = sparse.lil_matrix(sparse.csr_matrix(predict.tocsr()[:,0:trainingFeatureSize]))
    predict[:,firstSize:predict.shape[1]]=pow(10,-100)
    
    print(predict.shape[1])

    return predict

# Final method to start training including all other functions.
def training():

    training_fullCorpus , training_labels = createCorpus('Training')
    test_fullCorpus,test_labels = createCorpus('Testing')
    val_fullCorpus,val_labels = createCorpus('Validation')

    train , test, validate  = ngramFunc(training_fullCorpus,training_labels,test_fullCorpus,val_fullCorpus) # bag-of-words approach

    inputShape =train.shape[1:]
    print('------------- Input Shape ----------------')
    print(inputShape)
    adam = Adam(lr=0.0005) # Learning rate for this training.
    batchSize= 512 # Batch size 
    tensorboard = TensorBoard(log_dir="./logs",histogram_freq=0,write_graph=True,write_images=True) # Develop the tensorboard graph. Run tensorboard -logdir=./logs from current directory of this python file using terminal(MacOS/Linux) or CMD(Windows).
    earlystop = EarlyStopping(monitor='val_loss',patience=5) # To early stop the model before being over fitted.

    # Since this is a categorical classification (Because number of classes > 2) final layer softmax activation and categorical_crossentropy
    model = Sequential()
    model.add(Dense(64,activation='relu',input_shape=inputShape))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    model.add(Dropout(0.5))
    model.summary()

    if len(categories)>2:
        model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

        
    model.fit(train,training_labels,batch_size=batchSize,epochs=5000,callbacks=[tensorboard,earlystop],validation_data=(validate,val_labels))
    score, acc = model.evaluate(test,test_labels,batch_size=batchSize)
    print('Score: ',score) # Test Score 
    print('Accuracy: ',acc) # Test Accuracy (With this default configurations, 70.7% of testing accuracy)
    print('----------  Training Succesful ----------')
    model.save('Sinhala Text Classifier.h5') # Save model as HDF5 format.

def predict(txtfileName):
    currentLocation = os.path.abspath(__file__ + "/../")
    if not (os.path.isfile(currentLocation+'/Sinhala Text Classifier.h5')):
        print('Model is missing under the name of \"Sinhala Text Classifier.h5\". So training starts before predicting')
        training()
        
    else:
        pass
    print('Loading model. Please wait ... ') 
    model = load_model('Sinhala Text Classifier.h5')
    if os.path.abspath(__file__ + "/../../")+'/Predict/'+txtfileName+'.txt':
        filePath = os.path.abspath(__file__ + "/../../")+'/Predict/'+txtfileName+'.txt'
        inFile = open(filePath,'r')
        text = inFile.read().splitlines()
        predict = ngramFuncPrediction(text)
        print('Prediction starts')
        out = model.predict(predict)
        out = sum(out) / len(out)
        finalOut = (out > 0.5).astype(int)
        print(out)
        if finalOut[0] == 0:
            print('International')
        else : 
            print('Local')
    else:
        print('Prediction data included file not found')

predict('text')
        




