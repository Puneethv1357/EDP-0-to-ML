# 🎯 Emotion Classifier - Feel the Text!
 
# Overview

This marks our efforts in designing an NLP model that can take an input test and classify it into the six emotion classes, namely joy😊, sadness😢, fear😨, anger😠, surprise😲, love❤️. A team of three worked together on this project, dividing it into three different aspects: one dealing with the preprocessing, the second one dealing with model designing and training, and lastly one who creates the GitHub repo and deploys a web app with the designed model included in it. This project presents an emotion detection model built using a Bidirectional LSTM neural network in TensorFlow Keras. This model uses the [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) from Huggingface, preprocessed with spaCy for lemmatization and token filtering. A custom vocabulary is created, and input sequences are padded to a fixed length. The model includes embedding layers, stacked BiLSTM layers with dropout and L2 regularization, and a final dense layer with softmax for classification. The model is trained using categorical cross-entropy loss with early stopping and adaptive learning rate scheduling. Evaluation includes accuracy, precision, recall, a confusion matrix, and a classification report. Training history is visualized using accuracy and loss plots. The final trained model is saved as an (h5) file, and the vocabulary is saved as a pickle file for reuse. This solution is ideal for use in applications such as chatbots and emotional analysis systems.


# NLP Preprocess

About the preprocessing by the memeber of the team who dealt with it [A breifing about Preprocessing](https://docs.google.com/document/d/155r8F63NpeFQdOJJuy7jcSLpI0RqhAs02YyWSQBTnxo/edit?tab=t.0). This provides the reason behind all the preprocess we have do so far in this project with tabels indicating how the data has been manipulated to be abel to feed it to the model that was being designed. The token lengths across the three datasets, the training, validation, and test datasets, are as follows.


![Token lengths](https://github.com/Puneethv1357/EDP-0-to-ML/blob/da64b5109ced7cadf084392f6f9147c839cc68f0/images/Screenshot%202025-07-03%20175541.png)


The emotion class distribution across the total data used is as follows. 


![Class distribution](https://github.com/Puneethv1357/EDP-0-to-ML/blob/86409d940b51f171b79001fdcbe24773f5276487/images/emotion%20class%20distribution.png)

This concludes the preprocessing of the dataset that we chose. Shout out to Daksh Singh. His GitHub account is [here](https://github.com/D0905-ux). Check it out.

# Model design and training 
This is also one of the crucial parts of the project preprocessing, deciding what to do with it without a model, right ? . Here is what our team member has to say about [model designing and training](https://docs.google.com/document/d/1EOMEaN88uFxOhpIHro5SUKC20nWhUXK659GlZTBAy80/edit?addon_store&tab=t.0#heading=h.ilg8u4xwz13x), a dive into how it is structured and what the regulators used, dropout, and a short note on biLSTM. The validation accuracy and train accuracy are in the graph towards the left, whereas the train loss and validation loss graphs are on the right, with the number of epochs on the X-axis.

![graphs](https://github.com/Puneethv1357/EDP-0-to-ML/blob/0c86d4dabaadca4ba02e9e429bf0f5602b0d5e7d/images/plots.png)


This confusion matrix shows the model’s performance across six emotion classes. 


![confusion matrix](https://github.com/Puneethv1357/EDP-0-to-ML/blob/725e9405409ec78d12d3168b55b70b46778a7ce8/images/Confusion%20matrix.png)

This concludes the preprocessing of the dataset that we chose. Shout out to R. Poornashree. Her GitHub account is [here](https://github.com/Poornasshreee). Check it out.

# Web app design and deployment 

This is the part I dealt with, along with maintaining the GitHub repo that you are looking at right now. The web app is designed to be deployed on the Streamlit platform. It is nothing grand but a good-looking UI that takes input and produces output, along with a bar graph indicating the confidence of each emotion. A small guide on how to use this web [app](https://emotion-detector-0.streamlit.app/) is as follows 


In the first step, enter the text in the text box that is provided.


![entering test](https://github.com/Puneethv1357/EDP-0-to-ML/blob/48035de3199b40570c5997b846efbe4a27a8b79d/images/Entering%20text%20.png)


In the second step, click on the analyze button. 


![analyze](https://github.com/Puneethv1357/EDP-0-to-ML/blob/48035de3199b40570c5997b846efbe4a27a8b79d/images/click%20analyze.png)


VOILA!!! The prediction with the confidence bar graph is generated based on your input. 


![prediction](https://github.com/Puneethv1357/EDP-0-to-ML/blob/6254e9706940a04701bc9d6185ce6278987fceb4/images/prediction.png)


# Team and Contributions 
This project was developed as part of a collaborative team effort. The contributions are as follows:

Daksh Singh – Responsible for all data preprocessing steps, including text cleaning, tokenization, lemmatization, and vocabulary generation using spaCy.
Contact info
[github](https://github.com/D0905-ux)

R.Poornashree – Focused on model design and training, including BiLSTM architecture, hyperparameter tuning, and performance evaluation.
Contact info
[github](https://github.com/Poornasshreee)

V.Puneeth – Handled end-to-end integration, UI development using Streamlit, project documentation, and model deployment preparation.
Contact info
[github](https://github.com/Puneethv1357)

We worked together to build a complete and robust Emotion Detection system using deep learning and NLP.
