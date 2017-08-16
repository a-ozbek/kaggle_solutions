My work for the Planet - Amazon kaggle competition.  
[https://www.kaggle.com/c/planet-understanding-the-amazon-from-space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

* 40k total images. 32k/8k train/validation split.
* Finetuning Inception V3 CNN architecture in Keras framework
* Part 1: Replace the top fully-connected layer with a randomly initialized one then freeze all layers except this one. Train the top fully-connected layer a few epochs.
* Part 2: Unfreeze all layers and begin fine-tuning.
* Apply data augmentation (Random flips and rotations.)
* This solution ranked top 26% in the competition.

Bagging with 5-fold split could have increased the score significantly, but unfortunately training times take too long. 
