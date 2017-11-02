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
		self.first_layer_weights  = [[float(0) for x in range (0, n_in+1)] for y in range (0, n_hidden)]
		self.second_layer_weights = np.random.random_sample((n_out,n_hidden))
		self.prior_descent_first_layer_weights_deltas  = [[float(0) for x in range (0, n_in+1)] for y in range (0, n_hidden)]
		self.prior_descent_second_layer_weights_deltas = [[float(0) for x in range (0, n_hidden)] for y in range (0, n_out)]
		
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

		return
	
	def sigmoid(self, x):
		#return x
		return 1 / ( 1 + np.exp(-x))
		

	
	def compute_output(self, x_input):
		"""
		This method calculates the output of the neural network from the x_input.
		Sometimes this function is  called feed_forward
		We assume x_input is an array with dimension n_in.
		This will output an array of dimension n_out.
		The activation function for each layer is the sigmoid function - as per Mitchell.
		Other books (Bishop) use different activation functions for different layers.
		"""
		
		if (x_input.__len__() != self.n_in):
			raise ValueError ("wrong number of inputs")
		
		#
		# calculate the intermediary (hidden) step
		#
		hidden_units = [float for i in range (0, self.n_hidden)]
		for i in range (0, self.n_hidden):
			#the first of first_layer_weights is the weight corresponding to the threshhold - the additional constant x0 = 1
			#input is x1, x2, x3, ..., x_n_in
			s = self.first_layer_weights[i][0]
			for j in range (1, self.n_in+1):
				s += self.first_layer_weights[i][j] * x_input[j-1]
			s = s / self.adjust_for_sigmoid
			hidden_units [i] = self.sigmoid(s)
		
		#
		# calculate the outputs
		#
		output_units = [float for i in range (0, self.n_out)]
		for i in range (0, self.n_out):
			s = 0.0
			for j in range (0, self.n_hidden):
				s += self.second_layer_weights[i][j] * hidden_units[j]
			# adjust the parameter passed to the sigmoid function
			s = s / self.adjust_for_sigmoid
			output_units[i] = self.sigmoid(s)
		
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
		#  1.2 For each output unit calculate its error term: Delta(n_out) = O(n_out) * (1 - O(n_out) * (Train1 - O(n_out))
		#  1.3 For each hidden unit calculat its error term: Delta (n_hidden) = O(n_out) * (1 - O(n_out)) * (SUM (output_weights * Delta (n_out)
		#  1.4 For each output weight calculate: delta_weight (n_out) = learning_rate * Delta(n_out) * output_units (n_out)
		#  1.5 For each hidden weight calculate: delta_weight (n_hidden) = learning_rate * Delta(n_hidden) * hidden_units (n_hidden) + momentum * prior_delta_weight(n_hidden)
		# 2. Update each weight in the network 
		#  2.1 Update each hidden weight in the network: weight(n_hidden) = weight(n_hidden) + delta_weight(n_hidden)
		#  2.2 Update each prior_delta_weight to the current delta_weight (n_hidden)
		#  2.3 Update each output weight in the network: weight (n_out) = weight(n_out) + delta_weight (n_out)
		#  2.4 Update each prior_delta_weight to current delta_weight (n_out)
		#

		delta_output_weights = [[float(0) for x in range (0, self.n_hidden)] for y in range (0, self.n_out)]
		delta_hidden_weights = [[float(0) for x in range (0, self.n_in+1)] for y in range (0, self.n_hidden)]
		hidden_units = [float for i in range (0, self.n_hidden)]
		output_units = [float for i in range (0, self.n_out)]
		output_errors = [float for i in range (0, self.n_out)]
		hidden_errors = [float for i in range (0, self.n_hidden)]

		# 1. For each input in training example do
		for current_input_step in range (0, x_input.__len__()):
			#  1.1 Compute the output: O(n_out)
			input = x_input[current_input_step]
			if (input.__len__() != self.n_in):
				raise ValueError ("input vector not the same size as the neural network")
			
			#calculate the hidden (intermediary) units
			for i in range (0, self.n_hidden):
				#the first of first_layer_weights is the weight corresponding to the threshhold - the additional constant x0 = 1
				#input is x1, x2, x3, ..., x_n_in
				s = self.first_layer_weights[i][0]
				for j in range (1, self.n_in+1):
					s += self.first_layer_weights[i][j] * input[j-1]
				
				# adjust the parameter passed to the sigmoid function
				s = s / self.adjust_for_sigmoid
				hidden_units[i] = self.sigmoid(s)
			
			# calculate the output units
			for i in range (0, self.n_out):
				s = 0.0
				for j in range (0, self.n_hidden):
					s += self.second_layer_weights[i][j] * hidden_units[j]
				# adjust the parameter passed to the sigmoid function
				s = s / self.adjust_for_sigmoid
				output_units[i] = self.sigmoid(s)
		
			#  1.2 For each output unit calculate its error term: Delta(n_out) = O(n_out) * (1 - O(n_out) * (Train1 - O(n_out))
			for i in range (0, self.n_out):
				output_errors[i] = output_units[i] * (1-output_units[i])*(train[current_input_step][i]-output_units[i])
				#adjust the jacobian derivative for the scaling factor in the sigmoid function
				output_errors[i] = output_errors[i] / self.adjust_for_sigmoid
			
			#  1.3 For each hidden unit calculate its error term: Delta (n_hidden) = O(n_hidden) * (1 - O(n_hidden)) * (SUM (output_weights * Delta (n_out))
			for i in range (0, self.n_hidden):
				sum = 0.0
				for j in range (0,self.n_out):
					sum += self.second_layer_weights[j][i] * output_errors[j]
				hidden_errors[i] = hidden_units[i] * (1-hidden_units[i]) * sum
				#adjust the jacobian derivative for the scaling factor in the sigmoid function
				hidden_errors[i] = hidden_errors[i] / self.adjust_for_sigmoid
			
			#  1.4 For each output weight calculate: delta_weight (n_out) = learning_rate * Delta(n_out) * output_units (n_out) + momentum * prior_delta_weight(n_out)
			for i in range (0, self.n_out):
				for j in range (0,self.n_hidden):
					delta_output_weights[i][j] = self.learning_rate * output_errors[i]  * hidden_units[j] + self.momentum * self.prior_descent_second_layer_weights_deltas[i][j]
					
			
			#  1.5 For each hidden weight calculate: delta_weight (n_hidden) = learning_rate * Delta(n_hidden) * hidden_units (n_hidden) + momentum * prior_delta_weight(n_hidden)
			for i in range (0, self.n_hidden):
				#the first of first_layer_weights is the weight corresponding to the threshhold - the additional constant x0 = 1
				#input is x1, x2, x3, ..., x_n_in
				delta_hidden_weights[i][0] = self.learning_rate * hidden_errors[i] * 1 + self.momentum * self.prior_descent_first_layer_weights_deltas[i][0]
				
				for j in range (1, self.n_in+1):
					delta_hidden_weights[i][j] = self.learning_rate * hidden_errors[i] * input[j-1] + self.momentum * self.prior_descent_first_layer_weights_deltas[i][j]
			
			#
			# steps 1.1 - 1.5 use 7 loops. This can probably be done in 2 loops.
			# Think about that.....
			#

			# 2. Update each weight in the network 
			
			#  2.1 Update each hidden weight in the network: weight(n_hidden) = weight(n_hidden) + delta_weight(n_hidden)
			#  2.2 Update each prior_delta_weight to the current delta_weight (n_hidden)
			for i in range (0, self.n_hidden):
				for j in range (0, self.n_in+1):
					self.first_layer_weights[i][j] += delta_hidden_weights[i][j]
					self.prior_descent_first_layer_weights_deltas[i][j] = delta_hidden_weights[i][j]
			
			#  2.3 Update each output weight in the network: weight (n_out) = weight(n_out) + delta_weight (n_out)
			#  2.4 Update each prior_delta_weight to current delta_weight (n_out)
			for i in range (0, self.n_out):
				for j in range (0, self.n_hidden):
					self.second_layer_weights[i][j] += delta_output_weights[i][j]
					self.prior_descent_second_layer_weights_deltas[i][j] = delta_output_weights[i][j]
		
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

		self.first_layer_weights = fake_first
		self.second_layer_weights = fake_second
		
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