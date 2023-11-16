# Address-Split v.0.0.1
This is use BERT model in a Question Answering framework in order to split addresses into their constituents.

train.py > This loads the test-data.csv file and creates a squad type dataset using columns as questions and their constituent values as answers. Then it creates a question-answering based model using hugging face Transformers API and distilled-BERT encodings. 
          Model is then trained using the data in test-data.csv and saved in the Address-Split Folder.

main.py > The previously saved model is loaded and is used to split the address into constituents.


