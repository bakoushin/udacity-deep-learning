# Udacity Deep Learning Nanodegree Projects

List of projects I have built during [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) in 2020.

1. Bike Rental Patterns (MLP)
2. Dog Breed Classifier (CNN)
3. Style Transfer (CNN)
4. Generate TV Scripts (LSTM RNN)
5. Generate Human Faces (DCGAN)

The best way to explore these projects is to clone them in [Google Colab](https://colab.research.google.com/).

## Bike Rental Patterns (MLP)

[Jupyter Notebook](https://github.com/bakoushin/udacity-deep-learning/bike-rental-patterns/bike_rental_patterns.ipynb)

Very basic multi-layer perceptron which is trained on a [bike-sharing company dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) in order to predict trends in daily bike rental ridership.

Got good results on predicting daily rental patterns.

![Graph showing bike rental patterns](https://github.com/bakoushin/udacity-deep-learning/raw/master/images/bike-rental-patterns.png)

## Dog Breed Classifier (CNN)

[Jupyter Notebook: VGG16 from scratch vs. VGG16 pretrained (SGD)](https://github.com/bakoushin/udacity-deep-learning/dog-breed-classifier-2/dog_breed_classifier.ipynb)

[Jupyter Notebook: 3-layers CNN vs. VGG16 pretrained (Adam)](https://github.com/bakoushin/udacity-deep-learning/dog-breed-classifier-2/dog_app.ipynb)

The model detects dog breed from a given image. Fun feature: if accidentally a human is detected on the image, the app returns the most resembling dog breed.

![Images of dogs along with breed names](https://github.com/bakoushin/udacity-deep-learning/raw/master/images/dog-breed.png)

Comparison of two Convolutional Neural Networks in image classification task: the one trained from scratch and the one created using transfer learning.

Obviously, transfer learning won.

| Model                                                 | Accuracy |
| ----------------------------------------------------- | -------- |
| 3 convolutional layers model trained from scratch     | 19%      |
| VGG16 replica trained from scratch                    | 1%       |
| VGG16 transfer learning (Adam optimizer)              | 69%      |
| VGG16 transfer learning (SGD optimizer with momentum) | 86%      |

## Style Transfer (CNN)

[Jupyter Notebook](https://github.com/bakoushin/udacity-deep-learning/style-transfer/style_transfer.ipynb)

The model transfers the style of famous painting to a regular photograph based on the method outlined in [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf). The model leverages the fact that CNNs can extract content and style features separately. Thus they could be arbitrarily recombined.

Check out this cute kitten "by Van Gogh"!

![Kittens](https://github.com/bakoushin/udacity-deep-learning/raw/master/images/kitten.jpg)

## Generate TV Scripts (LSTM RNN)

[Jupyter Notebook](https://github.com/bakoushin/udacity-deep-learning/generate-tv-scripts/dlnd_tv_script_generation.ipynb)

RNN which generates TV-script based on a Seinfeld TV scripts dataset of scripts from 9 seasons. A set of LSTM cells which tries to predict the next word in the script based on the previous one. 11 different set of hyperparameters were tested to find out that the most modest ones work really well (little sentence length, smaller embeddings dimension, etc.).

Proof that computers could generate nonsense as good as humans.

> elaine: you know what? you don't have to talk to her.
>
> jerry: oh, i think i can get that.
>
> george: what? what?

## Generate Human Faces (DCGAN)

[Jupyter Notebook](https://github.com/bakoushin/udacity-deep-learning/generate-faces/dlnd_face_generation.ipynb)

A Deep Convolutional Generative Adversarial Network that could generate plausible human faces.

There are two networks: the first one, the Discriminator, which is strictly judging all provided images: are they either real or fake. The second one, Generator – a mastermind network, which tries to trick the Discriminator by generating human faces out of nowhere ("nowhere" is actually called "latent sample").

In memoriam of Dr. Frankenstein.

![Machine-generated human faces](https://github.com/bakoushin/udacity-deep-learning/raw/master/images/generated-faces.png)
