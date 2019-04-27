Deep Learning deals with learning data representations,  searching  for  useful  representations  of  some  input  data,  within  a  predefined  space  of  possibilities,  using  guidance from a feedback signal. 
The following machine learning models illustrate various concepts as I understand them. 


* **Project 1: Smoothers and Generalized Additive Models - predict the price of an Airbnb rental**

In the first part of this project, we explore 2 years Airbnb rental prices and fit several regression models for predicting the price given the booking time: polynomials, cubic B-splines, natural cubic splines. We tune hyper-parameters by cross-validation and compare models using R square.

In the second part, we explore listing features (address, room type, # beds, # bathrooms, etc.) and predict price using linear regression and polynomials.

In the third part, we explore geneartive additive models and smoothing splines.

We use R within the python environment, thanks to the rpy2 module. 

* **Project 2: Fully Connected Neural Networks - Classifying galaxies**

In this project, we work with real astronomical data from the Galaxy Zoo project with the purpose of classifying galaxies (elliptical and non-elliptical) based on visual attributes as they appear in astronomical surveys of the night sky, such as color and morphology. 

We explore the data and develop a method for outliers removal. We study how features affect the galaxy morphology using histograms. We exhibit data unbalancing, and create a baseline binary classifier as fully connected neural network using Keras and evaluate it using ROC-AUC. 
We use python imbalanced learning imblearn module to upsample the elliptical class. We investigate the effect of different learning rates on the Adam optimizer for this problem. We compare Adam and SGD using several values for momentum.

We also deal with the dataset shift problem, which appears when a model performs very well on test, but is poor when new data is used. We introduce approaches of solution such as the Kullback-Leibler Importance Estimation Procedure and transfer learning.

* **Project 3: Convolutional Neural Networks - Classifying gravitational waves from the LIGO observatory**

In this project we calculate convolutional operations, max pooling and average pooling operations by hand on example matrices. We carry our understanding on a simple RGB image by applying filters such as sharpen, blur.

We build a CNN for classifying CIFAR-10 images into the following classes: 'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'. We also solve interpretation, by visualizing the feature maps of convolution and max pooling layers. We also build saliency maps using gradiends of the output over the input to see which area of the image affects the classifier the most.

In the second part of this project, we consider the problem of classifying images as either gravitational waves or non-detection events using images from two detectors, one from the LIGO Hanford and LIGO Livingston. We build a CNN for this purpose and visualize its saliency maps.

* **Project 4: Recurrent Neural Networks - Named Entity Recognition**

In this project, we have a dataset with sentences, each composed of words with named entity tags (Tag) attached. We build several models to predict Tag using only the words themselves.

We explore frequency-based baseline model, vanilla feedforward neural network, recurrent neural network, gated recurrent neural network and bidirectional gated recurrent neural network.

We compare the models using F1 score. We also look at how the model if performing on each tag class. We develop methods for visualizing tags wrongly classified and we use data augmentation for improving the classification of those tags.

* **Project 5: Clustering and Reinforcement Learning - Hand posture recognition & Frozen Lake agent training**

In the first part of this project, we perform K-means clustering on records from a camera system performed on 14 different users performing 5 distinct hand postures. We work with R commands within python enviroment using rpy2, R stats module, factoextra library, fviz_cluster function. We evaluate the clusters using Gini coefficient, Entropy. We find the optimal number of cluster using elbow, Gap statistics,  silhouette and agglomerative clustering techniques.

In the second part, we work with OpenAI gym's FrozenLake environment. We train an agent to control the movement of a character in a grid world using value and policy iteration methods.

* **Project 6: Topic modeling & Bayesian analysis -  President Donald Trump's tweets analysis & Prediction of contraceptive usage by Bangladeshi Women**

In the first part of the project we process ca. 7000 tweets by President Donald Trump since his inauguration. We remove stop words, tokenize the tweets, calculate frequencies. We than build a Latent Dirichlet Allocation model for topics modeling and visualization.

In the second part, we predict women contraceptives usage based on region of residence, number of children and age. We build several Bayesian Logistic Regression, where the intercept and coefficients vary by district and prior distributions of normal and gamma is assumed. A  Markov chain Monte Carlo (MCMC) sampler is used to simulate data.

* **Project 7: Variational Autoencoder & Generative Adverserial Networks - Generating fake celebrity faces and fake anime faces**

Using a subset of the Celebrity dataset (202,599 number of face images) we train VAE and DCGAN models for facial generation. 

<hr>

*Disclaimer: the design and data of these projects are from the Advanced Topics in Data Science class at Harvard University - instructors: Pavlos Protopapas (SEAS), Mark Glickman (Statistics)*

