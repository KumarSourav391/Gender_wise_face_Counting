# Gender_wise_face_Counting

Introduction and Overview of Project
we're going to discover ways to construct a pc vision and deep learning aggregate utility. Which performs
gender smart face recognition with opencv and counts the people inside the photo or within the video.
In other words with the help of deep learning and computer vision algorithms using python
opencv as a modeling package , we will classify the gender and count the faces for a given
image/video.
These days deep getting to know field is one of the maximum revolutionary technologies with
rapid enhancement boom. It offers machines the ability to assume and analyze on their very
own. the important thing motivation for deep studying is to construct algorithms that mimic the
human brain.
you may have additionally heard about laptop imaginative and prescient. regularly abbreviated
as OpenCV. that is defined as a sub subject of look at that seeks to increase techniques to help
computers “see” and understand the content of digital images which includes snap shots and
videos.
2.1 Objectives of the project:
i. Predicting the gender of the presons.
ii. Counting the number of faces visible in image or in a video.
2.2 Models Used For Prediction:
1. Deep Convolutional Neural Network
2. OpenCV
2.3 Neural Networks:
A neural network is a series of algorithms that endeavors to recognize underlying
relationships in a set of data through a process that mimics the way the human
brain operates. In this sense, neural networks refer to systems of neurons, either
organic or artificial in nature. Neural networks can adapt to changing input; so the
network generates the best possible result without needing to redesign the output
criteria. The concept of neural networks, which has its roots in artificial intelligent,
is swiftly gaining popularity in the development of trending system.
Neural networks are multi-layer networks of neurons (the blue and magenta
nodes in the chart below) that we use to classify things, make predictions, etc.
The arrows that connect the dots shows how all the neurons are interconnected
and how data travels from the input layer all the way through to the output layer.
Advantages of Neural Network:
● Neural Networks have the ability to learn by themselves and produce
the output that is not limited to the input provided to them.
● The input is stored in its own networks instead of a data base, hence
the loss of data does not affect its working.
● These networks can learn from examples and apply them when a
similar event arises, making them able to work through real-time
events.
● Even if a neuron is not responding or a piece of information is missing,
the network can detect the fault and still produce the output.
● They can perform multiple tasks in parallel without affecting the system
performance.
1. Convolutional Neural Network
A Convolutional Neural Network (ConvNet/CNN) is a Deep mastering algorithm
that may soak up an input image, assign significance (learnable weights and
biases) to diverse components/objects inside the image and have the ability to
distinguish one from the opposite. The pre-processing required in a ConvNet is
tons decrease in comparison to different class algorithms. at the same time as in
primitive techniques filters are hand-engineered, with enough training, ConvNets
have the capacity to research those filters/characteristics.
The architecture of a ConvNet is analogous to that of the connectivity sample of
Neurons in the Human mind and became inspired by using the business
enterprise of the visual Cortex. person neurons respond to stimuli simplest in a
confined place of the visual field referred to as the Receptive area. a collection of
such fields overlap to cover the complete visual vicinity.
2. OpenCV
OpenCV is the huge open-source library for the computer vision, machine
learning, and image processing and now it plays a major role in real-time operation
which is very important in today’s systems. By using it, one can process images
and videos to identify objects, faces, or even handwriting of a human. When it
integrated with various libraries, such as NumPy, python is capable of processing
the OpenCV array structure for analysis. To Identify image pattern and its various
features we use vector space and perform mathematical operations on these
features.
OpenCV process
Advantages of OpenCV :
● First and foremost, OpenCV is available free of cost
● Since OpenCV library is written in C/C++ it is quite fast
● Low RAM usage (approx 60–70 mb)
● It is portable as OpenCV can run on any device that can run
Disadvantage of OpenCV :
● OpenCV does not provide the same ease of use when compared to MATLAB
● OpenCV has a flann library of its own. This causes conflict issues when you
try to use OpenCV library with the PCL library
3. Evaluation Metrics
3.1 Accuracy:
Accuracy is one metric for evaluating classification models.
Informally, accuracy is the fraction of predictions our model got right. Formally,
accuracy has the following definition:
For binary classification, accuracy can also be calculated in terms of positives and
negatives as follows:
Accuracy=TP+TNTP+TN+FP+FN
Where TP = True Positives, TN = True Negatives, FP = False Positives, and FN =
False Negatives.
3.2 Precision and Recall
Precision is a good measure to determine, when the costs of False Positive is
high. For instance, email spam detection. In email spam detection, a false positive
means that an email that is non-spam (actual negative) has been identified as
spam (predicted spam). The email user might lose important emails if the
precision is not high for the spam detection model.
In the field of information retrieval precision is the fraction of retrieved documents
that are relevant to the query:
Recall
Recall calculates how many of the Actual Positives our model capture through
labeling it as Positive (True Positive). Applying the same understanding, we know
that Recall shall be the model metric we use to select our best model when there
is a high cost associated with False Negative.
3.3 F1 Score
F1 is a function of Precision and Recall.
F1 Score might be a better measure to use if we need to seek a balance between
Precision and Recall AND there is an uneven class distribution (large number of
Actual Negatives) .
3.4 Support
Support is the number of actual occurrences of the class in the specified data
set.Imbalanced support in the training data may indicate structural weaknesses in
the reported scores of the classifier and could indicate the need for stratified
sampling or re-balancing.
3.5 Confusion matrix
A Confusion matrix is an N x N matrix used for evaluating the performance of a
classification model, where N is the number of target classes. The matrix
compares the actual target values with those predicted by the machine learning
mode l. This gives us a holistic view of how well our classification model is
performing and what kinds of errors it is making.
For a binary classification problem, we would have a 2 x 2 matrix as shown below
with 4 values:
Let’s decipher the matrix:
● The target variable has two values: Positive or Negative
● The columns represent the actual values of the target variable
● The rows represent the predicted values of the target variable
4. Data Preprocessing
Data preprocessing is a process of preparing the raw data and making it suitable
for a machine learning model. It is the first and crucial step while creating a
machine learning model.
When creating a machine learning project, it is not always a case that we come
across the clean and formatted data. And while doing any operation with data, it is
mandatory to clean it and put in a formatted way. So for this, we use data
preprocessing task.
A real-world data generally contains noises, missing values, and maybe in an
unusable format which cannot be directly used for machine learning models. Data
preprocessing is required tasks for cleaning the data and making it suitable for a
machine learning model which also increases the accuracy and efficiency of a
machine learning model.
Before going into pre-processing and data exploration we will explain some
of the concepts that allowed us t o select our features.
4.1 Exploratory Data Analysis :-
refers to the critical process of performing initial investigations on data so as
to discover patterns, to spot nomalies, to test hypo thesis and to check
assumptions with the help of summary statistics and graphical
representations .
4.1.1 Imports and Read In Data
4.1.2 loading dataset
4.1.3 Forming the DataFrame
4.2 EDA
It is a way of visualizing, summarizing and interpreting the information that is
hidden in rows and column format. EDA is one of the crucial step in data science
that allows us to achieve certain insights and statistical measure that is essential
for the business continuity, stockholders and data scientists. It performs to define
and refine our important features.
1. Handle Missing value
2. Removing duplicates
3. Outlier Treatment
4. Normalizing and Scaling( Numerical Variables)
5. Encoding Categorical variables( Dummy Variables)
6. Bivariate Analysis
4.2.1 Variable identification and data types
The very first step in exploratory data analysis is to identify the type of variables in
the dataset. Variables are of two types Numerical and Categorical. . dtypes method
to identify the data type of the variables in the dataset .
4.2.2 Size of the dataset
We can get the size of the dataset using the shape method.
4.2.3 describe the dataset
Describe() function to get various summary statistics that exclude NaN values.this
function returns the count, mean, standard deviation, minimum and maximum
values and the quantiles of the data.
4.3 Data preprocessing
4.3.1 Augmentation
After execution of this code, the independent variable X and dependent
variable Y will transform into the following.
4.3.2 Separating the dataset into test and train
Any machine learning algorithm needs to be tested for accuracy. In order to do
that, we divide our data set into two parts: training set and testing set. As the
name itself suggests, we use the training set to make the algorithm learn the
behaviours present in the data and check the correctness of the algorithm by
testing on testing set.
5. Model Building
The modeling process was divided into two main parts: traditional machine
learning models and deep neural networks. Simpler models were to be used as a
baseline for the convolutional neural network and recurrent neural network.
1.Convolution Neural Network (CNN)
we will build the Convolution Neural Network
● Conv2D : This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs. If use_bias is
True, a bias vector is created and added to the outputs. Finally, if
activation is not None , it is applied to the outputs as well.
● BatchNormalization: Batch normalization applies a transformation that
maintains the mean output close to 0 and the output standard deviation close
to 1.
● MaxPooling2D: Max pooling is a sample-based discretization process .
The objective is to down-sample an input representation (image, hidden-layer
output matrix, etc.), reducing its dimensionality and allowing for assumptions
to be made about features contained in the sub-regions binned.
● DropOut: The Dropout layer randomly sets input units to 0 with a frequency
of rate at each step during training time, which helps prevent overfitting.
Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all
inputs is unchanged.
After initializing we can now give the data to train the Neural Network.
Evaluating the model :
Classification report :
Confusion matrix for Gender Detection with Accuracy(94.61%) and
mainly this accuracy is comes when the model is not detecting more
than 10 images at a time.
Some sample outputs are :
● when we pass an image in our program :
![image](https://github.com/user-attachments/assets/cc1752f8-d49d-4b68-9dbb-8e5a9ff85398)


● Sample output from web cam :
![image](https://github.com/user-attachments/assets/dcd5c6e5-744f-448c-881a-52dc67281b28)

Conclusion
Machine learning (ML) methods has recently contributed very well in the
advancement of the prediction models used for energy consumption. Such
models highly improve the accuracy, robustness, and precision and the
generalization ability of the conventional time series forecasting tools. This project
demonstrates how we can leverage the Neural Networks to obtain the
classification of the gender of the people by detecting the images we provide and
also counts the number of persons available in the image. This system can be
employed in a variety of setups like in big companies or in various industrial
sectors to mark the attendance of the person by detecting his face, etc.
This project analysis the Gender of the people using Convolution Neural Network
with Accuracy(94.61%). The accuracy of the model can be increased by including
more images of the male and femal
Future scope
● To unlock mobile, without passcode.
● Google photos grouping the same person photos.
● Facial recognition in surveillance.
● Track attendence in any sector.
● Control access to sensitive areas.
Applications
● Amazon Just Go Technology
● Students Engagement
● Faster Payments
● Finding Missing Persons
Bibliography
1. Dataset
● https://dataaspirant.com/gender-wise-face-recognition-with-opencv
/
● https://www.kaggle.com/adityendrapba2021/face-counting-challen
ge/data
2. Documentation
● https://drive.google.com/drive/folders/1fG9IDEDkxlj_ESOSvRGeb
wnq147PJRiF
3. Resources
● https://dataaspirant.com/popular-activation-functions-neural-networ
ks/
● https://dataaspirant.com/popular-activation-functions-neural-networ
ks/
● https://towardsdatascience.com/face-detection-with-haar-cascade-
727f68dafd08
● https://opencv.org/
4. Algorithm
● https://dataaspirant.com/handle-overfitting-deep-learning-models/
● https://dataaspirant.com/ensemble-methods-bagging-vs-boostingdifference/
● https://towardsdatascience.com/a-comprehensive-guide-to-convol
utional-neural-networks-the-eli5-way-3bd2b1164a53
● https://towardsdatascience.com/a-complete-guide-to-principal-com
ponent-analysis-pca-in-machine-learning-664f34fc3e5a
