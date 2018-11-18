# neural-network-python
A neural network implemented from scratch in Python

## Reading List
* [Build a Neural Network](https://enlight.nyc/projects/neural-network/)
* [Implementing a Neural Network from Scratch in Python – An Introduction](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)
* [A Neural Network in 11 lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/)
* [How to build your first neural network with Python](https://medium.com/@UdacityINDIA/how-to-build-your-first-neural-network-with-python-6819c7f65dbf)
* [How to build your own Neural Network from scratch in Python](https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6)
* [How to build a three-layer neural network from scratch](https://medium.freecodecamp.org/building-a-3-layer-neural-network-from-scratch-99239c4af5d3)

## How To
The file *examples* currently includes five neural network examples:
* **exponential_sequence_example** - NN applied on an exponential sequence
* **normal_sequence_example** - NN applied on a normally distributed sequence
* **random_sequence_example** - NN applied on a random sequence
* **moons_decision_boundary_example** - Displays the decision boundary for a group 2D sequence
* **random_decision_boundary_example** - Displays the decision boundary for a random 2D sequence

The file also includes a decision boundary visualisation demo called **logistic_regression_decision_boundary_example** that uses Scikit-learn's regression fitting model.

### Neural network arguments
It is possible to change a number of arguments for the neural network:
* Input layer nodes
* Hidden layers and hidden nodes
* Output layer nodes
* Training iterations

#### Input layer nodes
By changing the number of input layer nodes you change how many inputs there are. For example, all sequences/series are 1D so the input layer should have one node. However, moons have *x* and *y* positions so they require two input nodes.

#### Hidden layers and hidden nodes
Hidden layers are in this implementation represented as an array of sizes. By adding a new number to the array you add a new hidden layer. These layers are placed between the input layer and the output layer. Changing a number in the array changes the number of nodes in that hidden layer. The number of layers and nodes required varies between problems and experimentation is often the key to success here.

#### Output layer nodes
By changing the number of output layer nodes you change how many outputs there are. For example, moons have two inputs (*x* and *y* poisitions), but they only have one output (a binary value, *0* or *1*).

#### Training iterations
You can change the number of traing iterations that the neural network should perform when training. The greater the number the more iterations it will perform which in turn hopefully makes the network better at predicting. One important thing to note is that it takes time to traing the neural network. So the greater the number, the longer it will take to train.

### Training
The neural network training function takes two arguments as input: input training data and output training data. Both arguments are specified as Numpy arrays/matrices. Columns are the input features and will be mapped to input layer nodes and rows are just input data entries. For example, the input matrix for moons should be two columns specifying the *x* and *y* positions and the rows should be the moon entries. The same thing applies for the output matrix, though in the moons example it should be a 1D matrix where each row is an expected value for a moon. Important to note is that all input and output should be scaled and only have values between *0* and *1*.

### Predicting
To predict is basically done as with the training, but without specifying an expected output. Remember to use the same number of features and use the same scale as for the training input. The output will also have to be scale, but this time scaled back to its original scale as the output from the predict function will be numbers in the range *0* to *1*.