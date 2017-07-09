# IML_Project
# Urdu Script Classification

## Data set
For training I made my own dataset by taking different samples of urdu script from different people. Taking the sample on different papers and crop it down and resize them accordingly. The dataset consists of approximately 759 images with the type of .jpeg. Then i label each of the character of urdu script individually and make 11 classes from it. Placing different samples of same letter in different class. Following are the 3 images of my dataset just as an example.

![3](https://user-images.githubusercontent.com/29806164/27997131-24fb1f4e-650b-11e7-8ce6-35f48e8b6849.jpg)
![4](https://user-images.githubusercontent.com/29806164/27997132-2535fcfe-650b-11e7-8ccb-726f79fb6613.jpg)
![1](https://user-images.githubusercontent.com/29806164/27997133-255db122-650b-11e7-84e8-21466cedc998.jpg)
![2](https://user-images.githubusercontent.com/29806164/27997134-25719066-650b-11e7-9364-adbd5a4ca462.jpg)

### Folder/Zipped files
Folder contains the following files
1. data cobtains all the images with their specified labels
2. tf_cnn_train contains the code to train your model using your dataset
3. tf_cnn_test contains the code to test your model using new images
4. model contains saved model

### Tool and language 
Using Anaconda tool 
Use Python Language with following pacakges
1. tensorflow
2. tflearn
3. h5py
4. hdf5
5. SciPy

### Algorithm / CNN layers
I am using CNN (Convolutional Neural Network) which specifiaclly used for images. I used multiple layers of cnn in my code.

network = conv_2d(network, 64, 3, activation='relu') 
network = max_pool_2d(network, 2) 
network = conv_2d(network, 32, 3, activation='relu')  
network = max_pool_2d(network, 2) 
network = conv_2d(network, 32, 3, activation='relu')  
network = max_pool_2d(network, 2) 
network = fully_connected(network, 256, activation='relu') 
network = dropout(network, 0.50) 
network = fully_connected(network, 256, activation='relu') 
network = dropout(network, 0.50) 
network = fully_connected(network, 11, activation='softmax') 
network = regression(network, optimizer='adam', 
loss='categorical_crossentropy', 
learning_rate=0.001) 

### Goal:
In my implementation, I used a Deep Learning approach to image recognition. Specifically, I leveraged the extraordinary power of Convolutional Neural Networks (CNNs) to recognize images. The task is to classify the different input images into letters of urdu language. CNNs are suitable for these type of problems. The main idea is that since there is a multiclass classification problem, we can construct the model in such a way that it would have an input size of a small training sample and a multi layer convolutional network at the top, which outputs the images and labels them for classification.

Essentially, this would be equal to:
1.	Cutting new big input image into squares of the models.
2.	Detecting the subject in each of those squares

### Model:
The model has been implemented in TensorFlow. The model may be trained directly from the Terminal invoking python model.py.

### How to run
You can run the model by typing "pyhton tf_cnn_train.py" in terminal, and check the accuracy of the model which in my case is about 97%.

### Results
Folowing is the screenshot of my result
![screenshot 3](https://user-images.githubusercontent.com/29806164/27997186-3d50e0c8-650c-11e7-96b6-5c6a730acd3f.png)

### Discussion:
I thoroughly studied the approach of applying SVM classifier but actually intended to employ the Deep Learning approach which clearly detects the images accordingly.
