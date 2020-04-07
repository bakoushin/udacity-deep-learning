# Udacity Deep Learning Nanodegree Projects

## Bike Rental Patterns (MLP)

[Jupyter Notebook](https://github.com/bakoushin/udacity-deep-learning/bike-rental-patterns/bike_rental_patterns.ipynb)

Very basic multi-layer perceptron which is trained on a [bike-sharing company dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) in order to predict trends in daily bike rental ridership.

Got good results on predicting daily rental patterns.

## Dog Breed Classifier (CNN)

[Jupyter Notebook: VGG16 from scratch vs. VGG16 pretrained (SGD)](https://github.com/bakoushin/udacity-deep-learning/dog-breed-classifier-2/dog_breed_classifier.ipynb)

[Jupyter Notebook: 3-layers CNN vs. VGG16 pretrained (Adam)](https://github.com/bakoushin/udacity-deep-learning/dog-breed-classifier-2/dog_app.ipynb)

The model detects dog breed from a given image. Fun feature: if accidentally a human is detected on the image, the app returns the most resembling dog breed.

Comparison of two Convolutional Neural Networks in image classification task: the one trained from scratch and the one created using transfer learning.

Obviously, transfer learning won.

| Model                                                 | Accuracy |
| ----------------------------------------------------- | -------- |
| 3 convolutional layers model trained from scratch     | 19%      |
| VGG16 replica trained from scratch                    | 1%       |
| VGG16 transfer learning (Adam optimizer)              | 69%      |
| VGG16 transfer learning (SGD optimizer with momentum) | 86%      |
