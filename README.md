# IML_Project
Urdu Script Recognition

Goal:
In my implementation, I used a Deep Learning approach to image recognition. Specifically, I leveraged the extraordinary power of Convolutional Neural Networks (CNNs) to recognize images. The task is to detect weather a given script is in urdu language or not.  CNNs are suitable for these type of problems. The main idea is that since there is a binary classification problem (urdu script/not), we can construct the model in such a way that it would have an input size of a small training sample and a single-feature convolutional layer of at the top, which output will be used as a probability value for classification.
Having trained this type of a model, the input's width and height dimensions can be expanded , transforming the output layer's dimensions to a map with an aspect ratio approximately matching that of a new large input.
Essentially, this would be equal to:
1.	Cutting new big input image into squares of the models.
2.	Detecting the subject in each of those squares

Data:
For training I made my own dataset by taking different samples of urdu script from different people. Taking the sample on different papers and crop it down and resize them accordingly. The dataset consists of approximately 759 images with the type of .jpeg. The i label each of the character of urdu script individually.
Model:
The model has been implemented in TensorFlow. The model may be trained directly from the Terminal invoking python model.py. You can run the model by typing "pyhton tf_cnn_train.py" in terminal, and check the accuracy of the model which in my case is about 97%.
Discussion:
I thoroughly studied the approach of applying SVM classifier but actually intended to employ the Deep Learning approach which clearly detects the images accordingly.
