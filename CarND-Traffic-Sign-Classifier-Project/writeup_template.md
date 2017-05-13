#**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./images/grayscale.png "Grayscaling"
[image3]: ./images/normalization.png "Normalization"
[image4]: ./images/1.png "Traffic Sign 1"
[image5]: ./images/2.png "Traffic Sign 2"
[image6]: ./images/3.png "Traffic Sign 3"
[image7]: ./images/4.png "Traffic Sign 4"
[image8]: ./images/5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth and fifth code cells of the IPython notebook.

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the values of each input are too large and want to make deviation to be small.

![alt text][image3]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| outputs 400        									|
| Fully connected		| outputs 200        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| outputs 100        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| outputs 43        									|
| Softmax				|         									|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eleventh cell of the ipython notebook.

To train the model, I used an adam optimizer, 128 batch size, and 20 epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the twelfth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.947
* test set accuracy of 0.925

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet is very common and it works pretty well in class, so I tried it first.
* What were some problems with the initial architecture?
The validation score is too low to pass the criteria.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Add dropout of 0.5 on fully-connected layers and increase the numbers of nodes in each layer.
* Which parameters were tuned? How were they adjusted and why?
The numbers of nodes in each layer because the model is under fitting.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution works well because there are many interesting shapes in the signs that are distinguish. Dropout works well because it reduces over fitting from many pixels of images.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
LeNet is very common and it works pretty well in class, so I tried it first.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
All scores are higher than 0.9


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because there is another sign nearby.
The second image might be difficult to classify because there is another sign nearby.
The third image might be difficult to classify because the background looks similar to the sign.
The fourth image might be difficult to classify because a lot of white colors.
The fifth image might be difficult to classify because most colors are white.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the fourteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop     			| Stop 										|
| Dangerous curve to the right					| Dangerous curve to the right											|
| Bumpy road	      		| Bumpy Road					 				|
| Turn left ahead			| Turn left ahead      							|
| Speed limit (70km/h)      		| Speed limit (70km/h)   									|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.925

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Stop sign (probability of 1.0), and the image does contain a Stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Stop   									|
| .00     				| No entry 										|
| .00					| Keep right											|
| .00	      			| Turn right ahead					 				|
| .00				    | Turn left ahead      							|


For the second image, the model is relatively sure that this is a Dangerous curve to the right sign (probability of 1.0), and the image does contain a Dangerous curve to the right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Dangerous curve to the right   									|
| .00     				| Children crossing 										|
| .00					| Slippery road											|
| .00	      			| Bicycles crossing					 				|
| .00				    | No passing      							|


For the third image, the model is relatively sure that this is a Bumpy road sign (probability of 0.99), and the image does contain a Bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Bumpy road   									|
| .01     				| Bicycles crossing 										|
| .00					| Road work											|
| .00	      			| Road narrows on the right					 				|
| .00				    | Traffic signals      							|


For the fourth image, the model is relatively sure that this is a Turn left ahead sign (probability of 1.0), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Turn left ahead   									|
| .00     				| Ahead only 										|
| .00					| Keep right											|
| .00	      			| Speed limit (60km/h)					 				|
| .00				    | Children crossing      							|


For the fifth image, the model is relatively sure that this is a Speed limit (70km/h) sign (probability of 1.0), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0         			| Speed limit (70km/h)   									|
| .00     				| Speed limit (30km/h) 										|
| .00					| Speed limit (20km/h)											|
| .00	      			| Speed limit (120km/h)					 				|
| .00				    | Speed limit (50km/h)      							|
