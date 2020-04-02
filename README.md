# Unsupervised Hierarchical Graph Representation Learning

Requirements:

pytorch>=0.41

tensorboardX==1.2

networkx==2.2

## Data set for Graph Classification

Evaluation script for various methods on [common benchmark datasets](http://graphkernels.cs.tu-dortmund.de) via 10-fold cross validation, where a training fold is randomly sampled to serve as a validation set. 

I uses the D&D data set as an example, for other datasets, they can be directly from the above websit. After downloading to the folder: ./data, and unzip it.

- COLLAB
- D\&D 
- PROTEINS 
- NCI-1


## Train model

To train the model, just run: 

./example.sh
