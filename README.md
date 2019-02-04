# Texture Encoder

## Purpose:

The purpose of this texture encoder is to create a many to one distillation of images using neural networks.
There are 2 modes to this. One is a residual network that adds a mask to the original image and calculates loss based on the new image,
and the other is a complete reconstruction of the image without a residual component to the network.
The goal is to create a mapping between many textures to one common texture to aid a reinforcement
 learning network in simulation to enable adaptability in transferring learning from one texture to that of another. All of the input
 images to this network are images captured from random coordinates on a proceduraly generated maze from the Unity Game Engine.


## Running Program:
Ensure that all the textured images are correctly placed in ./data.  There should be one texture
per folder with the same amount of images in each.  One folder should be named "real", this
will be the folder that contains the textures you would like generalized to.


Once this is in order, run "python3 main.py" to run the program.

## File Structure:

./:

	main.py:

		Adjust arguments as necessary. Mainly lays out arguments for model.

	modules.py:

		Contains elements required to create the architecture of the network such as
		"encoder", "decoder", and "residual block"

	transform_image.py:
		Similar to main.py, shows an example of how you could wrap this model to use in
		other code.  The example requires a weights file and displays an output image
		after undergoing an image transformation.

Checkpoints:

	Contains the weights files for resuming training or for loading wrapper function.

Models:

    Phi.py:

	   The code for class "phi" which will be called to train or load our entire model

Results:

	Generated_images [all] - the images generated from the encoder

	Masks - only valid for residual network, displays the masks created by the network

	Target - the target image

	Original[all] - the original image that was transformed

	Currently, the code will show outputs for the first 3 textures, more can be added if
	desired, but it should not be hard to do.

Utils:

	nnUtils.py:

		Contains 2 functions.  import_images is used to import all images as a numpy array
		from one directory.  import_images_ignore is used to import all images within
		all directories within a directory, excluding one specified directory.  In this
		example the excluded folder is labeled "real"
