# 🎯 Emotion Classifier - Feel the Text!
 
# Overview

This marks our efforts in desinging a nlp model that can take an input test and classify them into the six emotion classes namely joy😊,sadness😢,fear😨,anger😠,surprise😲,love❤️. A team of three worked together on this project dividing it into three different aspects one dealing with thee preprocessing , second one desling with model designing ad training and lastly one who creates the github repo and deploying a web app with the designed model included in it . This project presents an emotion detection model built using a Bidirectional LSTM neural network in TensorFlow Keras. This model uses the [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) from huggingface ,preprocessed with spaCy for lemmatization and token filtering. A custom vocabulary is created and input sequences are padded to a fixed length. The model includes embedding layers, stacked BiLSTM layers with dropout and L2 regularization, and a final dense layer with softmax for classfying .The model is trained using categorical cross entropy loss with early stopping and adaptive learning rate scheduling. Evaluation includes accuracy, precision, recall, confusion matrix, and a classification report. Training history is visualized using accuracy and loss plots. The final trained model is saved as an (h5) file and the vocabulary is saved as a pickle file for reuse.This solution is ideal for use in applications such as chatbots, and emotional analysis systems


# NLP Preprocess

About the preprocessing by the memeber of the team who dealt with it [A breifing about Preprocessing](https://docs.google.com/document/d/155r8F63NpeFQdOJJuy7jcSLpI0RqhAs02YyWSQBTnxo/edit?tab=t.0). This provides the reason behind all the preprocess we haqve do so far in this project with tabels indicating how the data has been manipulated to be abel to feed it to the model that was being designed. The token lenghts across three datasets the training , validation and test data set are as folllows


![Token lengths](https://github.com/Puneethv1357/EDP-0-to-ML/blob/da64b5109ced7cadf084392f6f9147c839cc68f0/images/Screenshot%202025-07-03%20175541.png)


And the emotion class distribution across the total data that is used is as follows 


![Class distribution](https://github.com/Puneethv1357/EDP-0-to-ML/blob/86409d940b51f171b79001fdcbe24773f5276487/images/emotion%20class%20distribution.png)
