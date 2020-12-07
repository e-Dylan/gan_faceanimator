# Face Animator AI (GAN)
GAN deep learning model to use AI generated faces from /gan_facegenerator, turns them into cartoon characters, and animates them.

### Functionality
This network was built to run off my previous [Face Generator](https://github.com/e-Dylan/gan_facegenerator) neural network which generates artifical images of faces.
These faces are then converted into cartoon-style graphics using latent vector projection and animated using the [first order animation model](https://github.com/AliaksandrSiarohin/first-order-model).

## Demo Face Cartoon/Animation
![Algorithm Demo](demo/face-animator-thumbnail.gif)

### Test This Network

You can test this program easily in my [Colab](https://github.com/e-Dylan/gan_cartoonizer/blob/master/gan_cartoonizer.ipynb) instance.

1. Open a Google Colab.
2. Paste each cell into a new code cell.
3. Edit -> Notebook Settings -> Hardware Accelerator: GPU.
4. Run first cell.
5. Setup data: Place your source face image you want to animate in stylegan2/raw directory - name it raw_face.jpg. Place your source animation video in the top level file directory in the left-side menu.
6. Run the cell.
