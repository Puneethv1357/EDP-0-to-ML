# 🎯 Emotion Classifier - Feel the Text!
 
# Overview

This marks our efforts in desinging a nlp model that can take an input test and classify them into the six emotion classes namely joy😊,sadness😢,fear😨,anger😠,surprise😲,love❤️. A team of three worked together on this project dividing it into three different aspects one dealing with thee preprocessing , second one desling with model designing ad training and lastly one who creates the github repo and deploying a web app with the designed model included in it . This project presents an emotion detection model built using a Bidirectional LSTM neural network in TensorFlow Keras. This model uses the [dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) from huggingface ,preprocessed with spaCy for lemmatization and token filtering. A custom vocabulary is created and input sequences are padded to a fixed length. The model includes embedding layers, stacked BiLSTM layers with dropout and L2 regularization, and a final dense layer with softmax for classfying .The model is trained using categorical cross entropy loss with early stopping and adaptive learning rate scheduling. Evaluation includes accuracy, precision, recall, confusion matrix, and a classification report. Training history is visualized using accuracy and loss plots. The final trained model is saved as an (h5) file and the vocabulary is saved as a pickle file for reuse.This solution is ideal for use in applications such as chatbots, and emotional analysis systems


# NLP Preprocess

About the preprocessing by the memeber of the team who dealt with it [A breifing about Preprocessing](https://docs.google.com/document/d/155r8F63NpeFQdOJJuy7jcSLpI0RqhAs02YyWSQBTnxo/edit?tab=t.0). This provides the reason behind all the preprocess we haqve do so far in this project with tabels indicating how the data has been manipulated to be abel to feed it to the model that was being designed. The token lenghts across three datasets the training , validation and test data set are as folllows


![Token lengths](https://github.com/Puneethv1357/EDP-0-to-ML/blob/da64b5109ced7cadf084392f6f9147c839cc68f0/images/Screenshot%202025-07-03%20175541.png)


And the emotion class distribution across the total data that is used is as follows 


![Class distribution](https://github.com/Puneethv1357/EDP-0-to-ML/blob/86409d940b51f171b79001fdcbe24773f5276487/images/emotion%20class%20distribution.png)

This concludes the preprocessing of the datasetr that we choose . Shout out to Daksh singh his github acoount is [here](https://github.com/D0905-ux) do check it out.

# Model design and training 
This is also one of the crucial parts of the project preprocessing done what to do with it without a model right ? . Here is what our team member has to say about [model desining and training](https://docs.google.com/document/d/1EOMEaN88uFxOhpIHro5SUKC20nWhUXK659GlZTBAy80/edit?addon_store&tab=t.0#heading=h.ilg8u4xwz13x) a dive into the how it is structured and what are the regulizers used dropout and short note on biLSTM. The validation accuracy and train accuracy is in the graph towards left where as the train loss and validation loss graph is on the right with no.of epochs on the X-axis

![graphs](https://github.com/Puneethv1357/EDP-0-to-ML/blob/0c86d4dabaadca4ba02e9e429bf0f5602b0d5e7d/images/plots.png)


This confusion matrix shows the model’s performance across six emotion classes 


![confusion matrix](https://github.com/Puneethv1357/EDP-0-to-ML/blob/725e9405409ec78d12d3168b55b70b46778a7ce8/images/Confusion%20matrix.png)

This concludes the preprocessing of the datasetr that we choose . Shout out to R.Poornashree her github acoount is [here](https://github.com/Poornasshreee) do check it out.

# Web app design and deployment 

This is the part i dealt with along with along with maintaining the github repo that you are looking at right now. The web app is desiged such that it can be deployed pn streamlit platformnohing grand but a good looking ui that takes a input and produces the output along with a bar graph indicting confidence of each emotion. A small guide on how to use this web app is ass follows 


First setp ennter the text in the text box that is provided to you


![entering test](https://github.com/Puneethv1357/EDP-0-to-ML/blob/48035de3199b40570c5997b846efbe4a27a8b79d/images/Entering%20text%20.png)


Second step click on the analyze button 


![analyze](https://github.com/Puneethv1357/EDP-0-to-ML/blob/48035de3199b40570c5997b846efbe4a27a8b79d/images/click%20analyze.png)


VOILA!!! The prediction with the the confidence bar graph are generated based on your input 


![prediction](https://github.com/Puneethv1357/EDP-0-to-ML/blob/48035de3199b40570c5997b846efbe4a27a8b79d/images/prediction.png)
