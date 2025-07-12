# 🎯 Emotion Classifier - Feel the Text!
 
# Overview

This marks our efforts in desinging a nlp model that can take an input test and classify them into the six emotion classes namely joy😊,sadness😢,fear😨,anger😠,surprise😲,love❤️. A team of three worked together on this project dividing it into three different aspects one dealing with thee preprocessing , second one desling with model designing ad training and lastly one who creates the github repo and deploying a web app with the designed model included in it . This project presents an emotion detection model built using a Bidirectional LSTM neural network in TensorFlow Keras. This model uses the [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) from huggingface ,preprocessed with spaCy for lemmatization and token filtering. A custom vocabulary is created and input sequences are padded to a fixed length. The model includes embedding layers, stacked BiLSTM layers with dropout and L2 regularization, and a final dense layer with softmax for classfying .The model is trained using categorical cross entropy loss with early stopping and adaptive learning rate scheduling. Evaluation includes accuracy, precision, recall, confusion matrix, and a classification report. Training history is visualized using accuracy and loss plots. The final trained model is saved as an (h5) file and the vocabulary is saved as a pickle file for reuse.This solution is ideal for use in applications such as chatbots, and emotional analysis systems


# NLP Preprocess

