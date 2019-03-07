# Unsupervised Hierarchical Graph Representation Learning

pytorch==0.41
tensorboardX==1.2

## Data set for Graph Classification

Evaluation script for various methods on [common benchmark datasets](http://graphkernels.cs.tu-dortmund.de) via 10-fold cross validation, where a training fold is randomly sampled to serve as a validation set.
Hyperparameter selection is performed for the number of hidden units and the number of layers with respect to the validation set: