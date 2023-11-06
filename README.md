# 💎 Custom Machine Learning Models From Scratch

> *"Or me, trying to understand greek alphabet and some obscure numbers in order to get a number close to 0.9"*

All of the models were written by **plugg1N** without any tutorials or guides. I've used some help getting formulae and some pieces of code, but I've never
watched a video or an article used to explain the process of creating a model from scratch. I've chosen to work on this project in order to comprehend
what happens under the hood of each model individually. This might help getting more accurate results or even boosting accuracy of models even further.

<p align="left">
<img src="https://img.shields.io/badge/PYTHON-black?style=for-the-badge&logo=python&logoColor=gold"/>
<img src="https://img.shields.io/badge/sklearn-black?style=for-the-badge&logo=scikitlearn&logoColor=gold"/>
<img src="https://img.shields.io/badge/JUPYTER-black?style=for-the-badge&logo=jupyter&logoColor=gold"/> </p>



# Linear Models

The simplest ones to implement and understand. I, probably, won't program ridge, 'cause it pretty much the same anyway. The hardest part of linear models is gradient boosting.
I am a 10th grader in Russian school on 1st semester, so we haven't learned about *derivatives or gradients* for that matter. So I had to impovise and get knowledge myself.

$$ \Large \theta_{t+1} = \theta_t - \eta \nabla L (f(x;\theta),y)$$

## Linear Regression

You can get source code from this [link](https://github.com/plugg1N/custom-ml-models/tree/main/linear-regression).

I had to start from scratch, so I've written a `simple.py` model. This is a model for simple Linear Regression task without Gradient Boosting whatsoever.
I needed to get my feet wet with custom models.

Next, I've written `linear_regression.py` module that was my last commit to that repo. Linear Regression module is simple and contains **96 lines of code**.
But it get the job done. Model can score, predict and train on given data of any shape.

If you are interested in my thinking process, you can look at my [drawing boards](https://github.com/plugg1N/custom-ml-models/tree/main/linear-regression/drawing-boards) of that directory.

*Here is the example:*

<img src="https://github.com/plugg1N/custom-ml-models/blob/main/linear-regression/drawing-boards/class_setup_%26_gradients.png" width=1280 height=500>

It was really hard to make, but possible. Thanks to some references and math snippets from the Internet.

Also, you can get **visualizations** of model. Here is one:

<p align="center"><img src="https://github.com/plugg1N/custom-ml-models/blob/main/linear-regression/visualizations/salary_data_visual.png"></p>


## Logistic Regression

You can get source code from this [link](https://github.com/plugg1N/custom-ml-models/tree/main/logistic-regression).

This model is pretty much the same as the first visually. Like, it is a linear function, but, for binary classification,
everything above the line - is class 1, below - class 0. Simple as! But under the hood, everything is damn different.

<p align="center"> <img src="https://1394217531-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-LvBP1svpACTB1R1x_U4%2F-Lw70vAIGPfRR1AjprLi%2F-LwAVc1EdfmPMge5dlYC%2Fimage.png?alt=media&token=d72e3231-0d64-4bb7-9e4c-20577940763d"> </p>

First of all, we use sigmoid activation function. Second of all, loss function is completely different. For our task, it is called `binary crossentropy`. It looks like this:

$$ \Large L_{CE}(y_{pred}, y) = - \frac{1}{m} \sum \limits_{i=1}^m y * log(y_{pred}) + (1-y) * log (1 - y_{pred}) $$

Much harder than this, right?

$$ \Large MSE = \frac {1} {m} \sum \limits_{i=1}^m (y - y_{pred})^2  $$

But getting gradients is simpler. Overall, logistic-regression implementation was way harder than I expected. I am greatful for finishing development. **The bad, thing is that**
accuracy is rather low on `breast_cancer` dataset and model works slower than expected. It is slower, because I was trying to get the job done, not to make
implementation better, than sklearn did.


