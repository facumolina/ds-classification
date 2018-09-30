# Data Structure Objects Classification

Using valid and invalid data structure objects, train a supervised learning algorithm in order to obtain a data structure object classifier.

## Getting started

After cloning the repository and installing the python dependencies, you can generate a classifier by running:

```bash
python3 gen-classifier.py datasets/singlylinkedlist-scope6.data datasets/singlylinkedlist-scope6.test
```

## Technical details

The gen-classifier.py script takes two arguments. The first one is the file containing the training set, while the second is the file containing the validation set. The training set will be used to train a feed forward neural network, which parameters are going to be determined by a random search. After that, a classification report will be displayed.



