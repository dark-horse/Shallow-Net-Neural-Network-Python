import sys
import os
import struct
import numpy as np

class neural_network (object):
	"""
	A structure to represent a neural network for solving Mitchell's Neural Networks for Face recognition problem.
	The main methods will be:
		Backpropagation - will train the network
		Compute (input) - given input X this will calculate the output of the network
	"""
	def __init__(self, learning_rate, momentum, n_in, n_hidden, n_out):
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.n_in = n_in
		self.n_hidden = n_hidden
		self.n_out = n_out

		#initialize the weights as per the algorithm in Mitchell section 4.7: An Illustrative Example: Face Recognition
		#First layer weights  initialized to 0.
		#Second layer weights are initialized to small random numbers. np.random.random_sample returns floats between -1 and 1
		#Prior deltas initialized to 0
		self.first_layer_weights  = np.zeros((n_hidden, n_in+1),dtype=np.float64)
		self.second_layer_weights = np.random.random_sample((n_out,n_hidden))

		self.prior_descent_first_layer_weights_deltas  = np.zeros((n_hidden,n_in+1),dtype=np.float64)
		self.prior_descent_second_layer_weights_deltas = np.zeros((n_out,n_hidden),dtype=np.float64)	
		
		# scaling factor for sigmoid
		# sigmoid function very quickly (at around 30) becomes equal to 1. If the output of the neural
		# network is 1, then there are no changes in the back propagation
		#(back propagation multiplies by output x (1-output). If output =1 , then 1-output = 0, and the whole 
		# multiplication is 0)
		# so we have to adjust the parameter passed to the sigmoid so the first pass (hidden nodes) does not get more
		# than 30.
		# because we adjust the parameter passed to the sigmoid function we also have to adjust the Jacobian derivatives
		# in the back propagation step.
		self.adjust_for_sigmoid = (n_hidden / 30) + 1
		
		# some array we will use over and over
		# allocate these at the beginning so we don't allocate on every back propagation
		# (we will be doing hundreds of back propagations)
		self.delta_output_weights = np.zeros((self.n_out,self.n_hidden),dtype=np.float64)
		self.delta_hidden_weights = np.zeros((self.n_hidden,self.n_in+1),dtype=np.float64)
		self.hidden_units = np.zeros(self.n_hidden,dtype=np.float64)
		self.output_units = np.zeros(self.n_out,dtype=np.float64)
		self.output_errors = np.zeros(self.n_out,dtype=np.float64)
		self.hidden_errors = np.zeros(self.n_hidden,dtype=np.float64)
		
		# some intermediary results arrays
		self.output_errors_intermediate = np.zeros(self.n_out,dtype=np.float64)
		self.hidden_errors_intermediate = np.zeros(self.n_hidden,dtype=np.float64)
		
		return
	
	def sigmoid(self, x):
		# x is a numpy array.
		# it would be great if there would be a version of numpy for vectorize / apply_along_axis 
		# 		which would do the calculation in place (instead of returning a new vector)
		# 		however, we can use Numpy primitives to calculate this in place ourselves
		
		#
		# the formula we want to get is 1 / ( 1 + np.exp(-x))
		#
		
		np.exp(-x, x)
		np.add(1, x, x)
		np.divide(1,x,x)
		
		# DONE. Now the input array is changed in place

	def compute_output(self, x_input):
		"""
		This method calculates the output of the neural network from the x_input.
		Sometimes this function is  called feed_forward
		We assume x_input is an array with dimension n_in+1 and the first parameter is 1 (this corresponds to the bias weight)
		This will output an array of dimension n_out.
		The activation function for each layer is the sigmoid function - as per Mitchell.
		Other books (Bishop) use different activation functions for different layers.
		"""
		
		if (x_input.shape[0] != self.n_in+1):
			raise ValueError ("wrong number of inputs")

		if ( abs(x_input[0] - 1.0) > 0.0000000001):
			raise ValueError ("the first parameter in the input should be 1 (the first weight is the bias weight)")
		
		#
		# calculate the intermediary (hidden) step
		#
		np.matmul(self.first_layer_weights, x_input, self.hidden_units)
		self.hidden_units.dot(1.0/self.adjust_for_sigmoid, self.hidden_units)
		self.sigmoid(self.hidden_units)
		
		#
		# calculate the outputs
		#
		output_units = np.matmul(self.second_layer_weights, self.hidden_units);
		output_units.dot(1.0/self.adjust_for_sigmoid, output_units)
		self.sigmoid(output_units)
		
		return output_units
	
	def back_propagation_full_gradient_descent(self, x_input, train):
		"""
		This function implements the neural network back propagation algorithm with
		full gradient descent (not the stochastic approximation to gradient descent)
		This is the algorithm used in Mitchell book (section 4.7 An Illustrative Example: Face Recognition)
		x_input is a collection of inputs (images) and train is a collection of observed outputs (left/right/straight/up directions)
		This function performs ONE gradient descent on the entire x_input
		The input is the matrix for the entire training example (maybe we should change this? It will take a lot of memory)
		"""
		
		#
		# ALGORITHM
		#
		# 1. For each input in training example do
		#  1.1 Compute the output: O(n_out)
		#  1.2 For each output unit calculate its error term: Delta(n_out) = O(n_out) * (1 - O(n_out)) * (Train1 - O(n_out))
		#  1.3 For each hidden unit calculat its error term: Delta (n_hidden) = O(n_hidden) * (1 - O(n_hidden)) * (SUM (output_weights * Delta (n_out))
		#  1.4 For each output weight calculate: delta_weight (n_out) = learning_rate * Delta(n_out) * hidden_units ()
		#  1.5 For each hidden weight calculate: delta_weight (n_hidden) = learning_rate * Delta(n_hidden) * input () + momentum * prior_delta_weight(n_hidden)
		# 2. Update each weight in the network 
		#  2.1 Update each hidden weight in the network: weight(n_hidden) = weight(n_hidden) + delta_weight(n_hidden)
		#  2.2 Update each prior_delta_weight to the current delta_weight (n_hidden)
		#  2.3 Update each output weight in the network: weight (n_out) = weight(n_out) + delta_weight (n_out)
		#  2.4 Update each prior_delta_weight to current delta_weight (n_out)
		#


		# 1. For each input in training example do
		for current_input_step in range (0, x_input.__len__()):
			input = x_input[current_input_step]

			if (input.shape[0] != self.n_in+1):
				raise ValueError ("input vector not the same size as the neural network. Expected {} and got {}".format(self.n_in_1, input.shape[0]))
			if (abs(input[0]-1.0)> 0.0000000001):
				raise ValueError ("Expected 1 for the first parameter in the input (the first weight is the bias weight). Got {} for step {}".format(input[0],current_input_step))

			#  1.1 Compute the output: O(n_out)
			#calculate the hidden (intermediary) units
			np.matmul(self.first_layer_weights, input, self.hidden_units)
			self.hidden_units.dot(1.0/self.adjust_for_sigmoid, self.hidden_units)
			self.sigmoid(self.hidden_units)
			
			# calculate the output units
			np.matmul(self.second_layer_weights,self.hidden_units,self.output_units)
			self.output_units.dot(1.0/self.adjust_for_sigmoid, self.output_units)
			self.sigmoid(self.output_units)
		
			#  1.2 For each output unit calculate its error term: Delta(n_out) = O(n_out) * (1 - O(n_out)) * (Train1 - O(n_out))
			np.subtract(1, self.output_units, self.output_errors)
			np.multiply(self.output_errors, self.output_units, self.output_errors)
			np.subtract(train[current_input_step], self.output_units, self.output_errors_intermediate)
			np.multiply(self.output_errors, self.output_errors_intermediate, self.output_errors)
			#adjust the jacobian derivative for the scaling factor in the sigmoid function
			self.output_errors.dot(1.0/self.adjust_for_sigmoid, self.output_errors)
			
			#  1.3 For each hidden unit calculate its error term: Delta (n_hidden) = O(n_hidden) * (1 - O(n_hidden)) * (SUM (output_weights * Delta (n_out))
			#  we want to use numpy, and avoid allocating a new array.
			np.subtract(1, self.hidden_units, self.hidden_errors)
			np.multiply(self.hidden_units, self.hidden_errors, self.hidden_errors)
			np.matmul(self.output_errors, self.second_layer_weights, self.hidden_errors_intermediate)
			np.multiply(self.hidden_errors_intermediate, self.hidden_errors, self.hidden_errors)
			#adjust the jacobian derivative for the scaling factor in the sigmoid function
			self.hidden_errors.dot(1.0/self.adjust_for_sigmoid, self.hidden_errors)

			#  1.4 For each output weight calculate: delta_weight (n_out) = learning_rate * Delta(n_out) * hidden_units () + momentum * prior_delta_weight(n_out)
			np.outer(self.output_errors, self.hidden_units, self.delta_output_weights)
			self.delta_output_weights.dot(self.learning_rate, self.delta_output_weights)
			self.prior_descent_second_layer_weights_deltas.dot(self.momentum, self.prior_descent_second_layer_weights_deltas)
			np.add(self.delta_output_weights, self.prior_descent_second_layer_weights_deltas, self.delta_output_weights)
			
			#  1.5 For each hidden weight calculate: delta_weight (n_hidden) = learning_rate * Delta(n_hidden) * input () + momentum * prior_delta_weight(n_hidden)
			np.outer(self.hidden_errors, input, self.delta_hidden_weights)
			self.delta_hidden_weights.dot(self.learning_rate, self.delta_hidden_weights)
			self.prior_descent_first_layer_weights_deltas.dot(self.momentum, self.prior_descent_first_layer_weights_deltas)
			np.add(self.delta_hidden_weights, self.prior_descent_first_layer_weights_deltas, self.delta_hidden_weights)
					
			#
			# steps 1.1 - 1.5 use 7 loops. This can probably be done in 2 loops.
			# Think about that.....
			#

			# 2. Update each weight in the network 
			
			#  2.1 Update each hidden weight in the network: weight(n_hidden) = weight(n_hidden) + delta_weight(n_hidden)
			#  2.2 Update each prior_delta_weight to the current delta_weight (n_hidden)
			np.add(self.first_layer_weights, self.delta_hidden_weights, self.first_layer_weights)
			np.copyto(self.prior_descent_first_layer_weights_deltas, self.delta_hidden_weights)
			
			#  2.3 Update each output weight in the network: weight (n_out) = weight(n_out) + delta_weight (n_out)
			#  2.4 Update each prior_delta_weight to current delta_weight (n_out)
			np.add(self.second_layer_weights, self.delta_output_weights, self.second_layer_weights)
			np.copyto(self.prior_descent_second_layer_weights_deltas, self.delta_output_weights)
		
		return
	
	def calculate_error(self, x_input, train):
		"""
		this returns the error of the neural network for the given input.
		The error is the euclidian distance squared
		"""
		res = 0.0
		for i in range (0, x_input.__len__()):
			y = self.compute_output(x_input[i])
			for j in range(0, self.n_out):
				res = (y[j] - train[i][j])**2
		return res
	
	def fake_back_propagation(self):
		"""
		Fake neural network to test some matrix definitions and matrix multiplications
		"""
		fake_first = [[float for x in range (0,self.n_in+1)] for y in range (0,self.n_hidden)]
		fake_second = [[float for x in range (0,self.n_hidden)] for y in range (0,self.n_out)]

		for i in range (0,self.n_hidden):
			for j in range (0,self.n_in+1):
				fake_first [i][j] = 0.1 * (i + 1) + 0.1 * (j+1)
		for i in range (0,self.n_out):
			for j in range (0,self.n_hidden):
				fake_second [i][j] = 0.01 * (i + 1) + 0.01 * (j+1)

		self.first_layer_weights = np.add(self.first_layer_weights, fake_first)
		self.second_layer_weights = np.zeros((self.n_out,self.n_hidden))
		self.second_layer_weights = np.add(self.second_layer_weights, fake_second)
		
		return fake_first, fake_second
		
	def save_neural_network(self, save_fn):
		"""
		This method will save the current neural network to disk.
		The format is:
		first two bytes are DH
		then follows the learning rate in floating point notation.
		And then one comma
		then follows the momentum in floating point notation
		ANd then one comma
		next 4 bytes are the number of inputs (n_in) in little-endian (INTEL) notation. n_in does not include the additional bias weight
		next 4 bytes are the number of hidden (n_hidden) in little-endian (INTEL) notation.
		next 4 bytes are the number of outputs (n_out) in little-endian (INTEL) notation.
		then follows comma separated n_in x n_hidden weights (first layer weights) in floating point notation.
		then follows comma separated n_hidden x n_out weigts (second layer weights) in floating point notation.
		Right now we will not save the previous weights (the ones we use in the momentum calculation).
		"""
		f = open(save_fn, "wb")
		
		# first two bytes are DH
		f.write("DH")
		# learning rate in floating point notation. Use 4 decimals
		f.write("{0:.4}".format(self.learning_rate))
		#and then a comma
		f.write(",")
		#momentum in floating point notation. Use 4 decimals
		f.write("{0:.4}".format(self.momentum))
		#and then a comma
		f.write(",")
		#next four bytes is n_in in little-endian
		f.write(struct.pack("i", self.n_in))
		#next four bytes are n_hidden in little-endian
		f.write(struct.pack("i", self.n_hidden))
		#next four bytes are n_out in little-endian
		f.write(struct.pack("i", self.n_out))	
		#now write the first_layer weigths in float with 15 digits precision comma separated
		for i in range (0, self.n_hidden):
			weight = self.first_layer_weights[i]
			for j in range(0, self.n_in+1):
				f.write("{0:.15f}".format(weight[j]))
				f.write(",")
		
		#now write the second layer weights in float with 15 digits precision comma separated
		for i in range (0, self.n_out):
			weight = self.second_layer_weights[i]
			for j in range (0, self.n_hidden):
				f.write("{0:.15f}".format(weight[j]))
				f.write(",")
		#and we are done
		f.close()
		return
	
	def open_neural_network(self, open_fn):
		f = open (open_fn, "rb")
		cur = f.read(2)
		if (cur[0] != "D" or cur[1] != "H"):
			raise ValueError ("Format unknown")
		new_learning_rate = 0.0
		new_momentum = 0.0
		new_n_in = 0
		new_n_hidden = 0
		new_n_out = 0
		
		#learning rate
		buf = ""
		cur = f.read(1)
		while cur != ",":
			buf = buf + cur
			cur = f.read(1)
		new_learning_rate = float(buf)
		
		#momemtum
		buf = ""
		cur = f.read(1)
		while cur != ",":
			buf = buf + cur
			cur = f.read(1)
		new_momentum = float (buf)
		
		#n_in
		buf = f.read(4)
		new_n_in = struct.unpack("i", buf)[0]
		
		#n_hidden
		buf = f.read(4)
		new_n_hidden = struct.unpack("i", buf)[0]
		
		#n_out
		buf = f.read(4)
		new_n_out = struct.unpack("i", buf)[0]
		
		new_nn = neural_network(new_learning_rate, new_momentum, new_n_in, new_n_hidden, new_n_out)
		
		#first layer weights
		for i in range (0, new_n_hidden):
			for j in range (0, new_n_in+1):
				buf = ""
				cur = f.read(1)
				while (cur != ","):
					buf = buf + cur
					cur = f.read(1)
				new_nn.first_layer_weights[i][j] = float(buf)
		
		# second layer weights
		# the neural network constructor initializes the second layer weigts to small random values.
		# we will overwrite them with what we read in the file
		for i in range (0, new_n_out):
			for j in range (0, new_n_hidden):
				buf = ""
				cur = f.read(1)
				while (cur != "," and cur != ""):
					buf = buf + cur
					cur = f.read(1)
				new_nn.second_layer_weights[i][j] = float(buf)
		
		# and we are done!
		f.close()
		return new_nn
	
	
	def identical_neural_networks (self, compare):
		if (self.learning_rate != compare.learning_rate):
			print "different learning rates: self is {} and compare is {}.".format(self.learning_rate, compare.learning_rate)
			return False
		
		if (self.momentum != compare.momentum):
			print "different momentums: self is {} and compare is {}.".format(self.momentum, compare.momentum)
			return False
		
		if (self.n_in != compare.n_in):
			print "different n_in: self is {} and compare is {}.".format(self.n_in, compare.n_in)
			return False
		
		if (self.n_out != compare.n_out):
			print "different n_out: self is {} and compare is {}.".format(self.n_out, compare.n_out)
			return False
		
		if (self.n_hidden != compare.n_hidden):
			print "different n_hidden: self is {} and compare is {}.".format(self.n_hidden, compare.n_hidden)
			return False
		
		for i in range (0, self.n_hidden):
			for j in range (0, self.n_in + 1):
				if (abs(self.first_layer_weights [i][j] - compare.first_layer_weights [i][j]) > 0.000000000000001):
					print "different first layer weights for row {} and column {}. self has {} and compare has {}.".format(i,j,self.first_layer_weights[i][j], compare.first_layer_weights[i][j])
					return False
		
		for i in range (0, self.n_out):
			for j in range (0, self.n_hidden):
				if (abs(self.second_layer_weights [i][j] - compare.second_layer_weights [i][j]) > 0.000000000000001):
					print "different second layer weights for row {} and column {}. self has {} and compare has {}.".format(i,j,self.second_layer_weights[i][j], compare.second_layer_weights[i][j])
					return False

		return True