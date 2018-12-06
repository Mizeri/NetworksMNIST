# Libraries
import mnist_loader
import network

# Defining 'training_data', 'validation_data', 'test_data', see mnist_loader.py
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

'''Create a network with 3 layers: an input layer(28 * 28 = 784 input neurons),
a hidden layer(30 neurons), an output layer(10 neurons representing 10 digits)
'''
net = network.Network([784, 32, 10])

'''Stochastic gradient descent algorithm, 30 epochs, mini_batch_size = 10,
eta = 3.0
'''
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
