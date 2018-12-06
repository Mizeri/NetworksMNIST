"""
Exercise: Try creating a network with just two layers - an input and an output
layer, no hidden layer - with 784 and 10 neurons, respectively. Train the
network using stochastic gradient descent. What classification accuracy can you
achieve?
"""
import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 10])
net.SGD(training_data, 15, 10, 2.0, test_data=test_data)
