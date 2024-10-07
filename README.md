# cv-xavier-reese

### Part 1: Conceptual Design
My semester project idea is a program that recognizes trees on campus from images. If possible, I will expand it to recognizing a wider range of trees, say those found in Indiana or Northeast US. I like this project especially because I would enjoy and use a program like that myself. In order to train, validate, and test my model I will need a dataset of images. I have seen some good datasets online in limited research for leaves, and trees are not so unique that I think I will find trouble with getting large datasets of any image type that I choose. However, I will have to decide what types of images to use. I think with the use case of this program, I can require that all of the photos have the tree shown predominantly in the foreground, as people typically want to identify a tree that they are in front of in real life and can take a quality photo of in the moment. I will need to make sure that the program doesn't find difficulty in identifying the tree over a variety of backgrounds. If this proves too difficult, I could also pivot and focus on identifying trees by their leaves, which would allow for a better control of the background of photos taken.

To identify trees from a zoomed-out photo, things like overall shape, bark pattern and color, branch structure, leaf density, and more could be used as features for the program. There are some things that change with time for a tree, like leaf colors changing in the fall and leaves falling off in the winter. Depending on the final intended scope of my project, I can either expand it to include winter and fall photos or only summer photos. I know that it could considerably complicate the training of my model to include seasonal photos, but I don't know if it will make it ineffective. If I find good datasets for both options, I can compare the two models at different scopes of tree recognition.

No matter the scope of the program, I will have to detect the tree or leaf within the frame, and differentiate it from the background. The backdrop will often be trees or other plants, similar in color and shapes to the tree. The program will have to learn to ignore vegetation that is farther from the camera, blurrier or identifiable as not a tree by other characteristics (low to the ground, etc.)

Finally, I will tackle the program from both a CNN-based and non-deep-learning based approach and compare them.

So overall, this project will be about collecting a variety of datasets of both trees and possibly leaves, to analyze the power of computer vision to classify them during different seasons, and with larger or smaller geographical scopes. I don't expect to be able to write a program that recognizes world or country-wide trees without geographical input, but I'm interested to see how large I can push the scope of the program to be while maintaining a useful level of accuracy.

### Part 2: Data Acquisition

I found a database of images of 12 different types of leaves, with or without different diseases. It includes, however, a healthy dataset for each with 1000+ images. I will simply split these in a 60/20/20 fashion to create my training/testing/unknown datasets. This way there are no meaningful differences between the datasets for now.

https://data.mendeley.com/datasets/tywbtsjrjv/1

The photos are taken with a grey backgound, making the leaves very easy to distinguish. They are taken in high resolution and well-lit, although not always uniform. The database also uses six different augmentation techniques for increasing the data-set size. The techniques are image flipping, Gamma correction, noise injection, PCA color augmentation, rotation, and Scaling.

This dataset will allow me to create a model that recognizes leaves and has to account for differences in leaf structure. It also includes a folder of background images which don't include leaves, I assume so you can start to have your model recognize leaves over top of common backgrounds (cities, general greenery, dirt, roads, etc.) I'm not sure that this class will cover the skills needed to utilize this.

My original plan (see above) was to recognize trees on campus. I couldn't find a useful dataset of images for these specific tree species, so for now I am pivoting to the leaves that lend themselves to better datasets.

If my final goal remains to identify trees on campus, I would like to get a better dataset for my needs in the future. I plan to talk with Professor Czajka about just how many photos I would need to take of leaves from trees on campus. If it is possible, I'll take those images myself and then tweak my model to work for the new dataset, or perhaps take the photos before I even create the model.

