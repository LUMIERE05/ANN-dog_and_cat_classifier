# ANN-dog_and_cat_classifier
Artificial Neural Network to classifies dog and cat, then display it to GUI and a robot using camera

This is an image classifier using ANN for learning algorithm. It will choose what picture got displayed to camera, it will be dog or cat.
The preprocessing that being used in this image classifier is ORB(Oriented FAST and Rotated BRIEF) and PCA. The ORB will find unique value in a picture. That value will be become input data for ANN. Input data for ANN consist with 2 value, ORB with high nfeature value and ORB with low nfeature value. High or low the nfeature that being used will determine how much unique value can be found in a picture. So the higher nfeature being used the higher chance to found much more unique value. After ORB data obtained, the shape of ORB data will got reshaped to (-1, 1) dimension and got scaled using Standard Scaler and got transform using PCA. And then that data will got cut so the remaining data is 100. From that data, we take the mean so we can have final data for input data ANN. The process will always repeat itself until there is no more picture is going to use. After all pictures was processed, we normalize the data using MinMax Scaler equation for respective type data.

Input data for training are 2000 pictures including 1000 cat pictures and 1000 dog images. For testing are 500 pictures including 250 cat pictures and 250 dog pictures. There are two types pictures that being used, pictures with background and pictures without background. We can choose which training data we want to use, and testing data we want to use. For example for training we use pictures with background and for testing we use pictures without background.

ANN algorithm is construct mainly using NUMPY and PANDAS. That being said, this ANN is not using Tensorflow. The reason why using JUST numpy and pandas is to get better understanding what happen behind learning process, how to implement activation function, derivative of activation function, and matrix.

The order of use of this coding:
1. You need to run python file name "FOR MAKING DATA INPUT.py" to make data input and store it to respective folder
2. You need to run python file name "FOR MAKING TESTING DATA.p" and "FOR MAKING TRAINING DATA.py"
3. You need to run python file name "FEEDFORWARD BACKPROPAGATION.py" to start learning algorithm
4. You need to run python file name "FEEDFORWARD.py" to find best configuration
5. You need to run python file name "main.py"

note:
you can use arduino or similar to make robot using data from ANN

Dataset's source: https://github.com/laxmimerit/dog-cat-full-dataset

I hope i can help you with my code :)
