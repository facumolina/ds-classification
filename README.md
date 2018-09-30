# Data Structure Object Classification

Using valid and invalid data structure objects, train a supervised learning algorithm in order to obtain a data structure object classifier.

## Getting started

After cloning the repository and installing the python dependencies, you can generate a classifier by running:

```bash
python3 gen-classifier.py datasets/singlylinkedlist-scope6.data datasets/singlylinkedlist-scope6.test
```

You can use any file matching datasets/filename.data as training set and the corresponding file datasets/filename.test as the validation set

## Technical details

The gen-classifier.py script takes two arguments. The first one is the file containing the training set, while the second is the file containing the validation set. The training set will be used to train a feed forward neural network, which hyperparamters parameters are going to be determined by a random search. After that, a classification report will be displayed.

## Application

Let's consider the simple data structure Singly Linked List and the following Java implementation:

<img width="278" alt="captura de pantalla 2018-09-30 a la s 20 13 10" src="https://user-images.githubusercontent.com/7095602/46263993-578dbc80-c4ed-11e8-9a98-ca2807f720c7.png">

Valid objects are those that satisfy the class invariant of Singly Linked List, and invalid objects are those that do not satisfy the invariant. The class invariant states that the list must be acyclic and that the number of nodes must equal to the value of the size field:

<img width="401" alt="captura de pantalla 2018-09-30 a la s 20 14 55" src="https://user-images.githubusercontent.com/7095602/46264011-860b9780-c4ed-11e8-9021-8e058cda1977.png">

Commonly, class invariants are not present in the data structures implementations and thus some program analysis tools can't be fully exploited. So, if you are able to obtain an adequate dataset containing valid and invalid objects of some data structure, a neural network can be trained in order to approximate the real class invariant, and then enabling certain kind of automated analysis.

