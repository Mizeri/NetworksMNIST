# Libraries
import mnist_loader
import network

# Defining 'training_data', 'validation_data', 'test_data', see mnist_loader.py
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

'''Create a network with 3 layers: an input layer(28 * 28 = 784 input neurons),
a hidden layer(30 neurons), an output layer(10 neurons representing 10 digits)
'''
<<<<<<< HEAD
net = network.Network([784, 32, 10])
=======
net = network.Network([784, 30, 10])
>>>>>>> 8394027f01a3bde77f97ebf9f0dc80fcf47c62b4

'''Stochastic gradient descent algorithm, 30 epochs, mini_batch_size = 10,
eta = 3.0
'''
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
