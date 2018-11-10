# PreTrainingCNN
Sample Code for Pre training convolutional neural networks via autoencoders


Initial idea came while studying variational autoencoders. Typical deep CNN's can take many epochs to converge to a reasonable accuracy level. I decided to attempt to decrease number of epochs required to train the CNN, by placing the final classifier, as the intermediate layer of convolutional autoencoder. A loss function was then created determined as a linear combination of the cross entropy on the intermediate layer, plus the KL divergence of the final result with the inital image. 

Testing was performed on MNIST data. I did not force the layers of the autoencoders to have the same weights, though this is a typical requirement.

Tests show the convolutional autoencoder converged to within 90% after 3 epochs, while the CNN remained at 13% accuracy.

Later found out this had already been published in a paper (https://www.ni.tu-berlin.de/fileadmin/fg215/teaching/nnproject/cnn_pre_trainin_paper.pdf)
