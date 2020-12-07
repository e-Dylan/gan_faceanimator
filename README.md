# Face Animator AI (GAN)
GAN deep learning model to use AI generated faces from /gan_facegenerator, turns them into cartoon characters, and animates them.

### Functionality
This network was built to run off my previous [Face Generator](https://github.com/e-Dylan/gan_facegenerator) neural network which generates artifical images of faces.
These faces are then converted into cartoon-style graphics using latent vector projection and animated using the [first order animation model](https://github.com/AliaksandrSiarohin/first-order-model).

## Demo Face Cartoon/Animation
![Algorithm Demo](demo/face-animator-thumbnail.gif)

### Test This Network

You can test this program easily in my [Colab]() instance.

The models have already been trained, there is no need to train them.

## Training progress visualization.
![App Demo](demo/training_visual.gif)

## Training Process
15-minute training loss graph for 64 x 64 groundtruth images.
![Train Loss Graph](train-loss-graph.png)

Training is done by feeding 64-sized batches of face images using the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. Images were scaled to 128x128 and the network architecture was designed accordingly. The network learns to extract features from human faces and replicate them artificially by gradient descent.

Training was done on a single GPU for roughly 3 hours. The final product generates believable human faces at 128x128 resolution. These are visible at [/demo](https://github.com/e-Dylan/gan_facegenerator/tree/master/demo)
