import neural_network as nn
import read_input as ri
import time

#
# the digit target vectors (these are constant)
#
zero 	= [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
one 	= [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
two 	= [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
three 	= [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
four	= [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
five	= [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]
six 	= [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
seven 	= [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
eight 	= [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1]
nine 	= [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9]

digit_vector = [zero, one, two, three, four, five, six, seven, eight, nine]

def digit_recognition():
	"""
	This method will train the neural network and then test it on the test sample.
	"""
	
	train_fn = "..\\train-images-idx3-ubyte"
	train_target_fn = "..\\train-labels-idx1-ubyte"
	
	test_fn = "..\\t10k-images-idx3-ubyte"
	test_target_fn = "..\\t10k-labels-idx1-ubyte"
	
	#read the images
	train = ri.read_images(train_fn)
	# scale down the digits read because with lots of hidden nodes the output of the function is 1 (due to sigmoid function)
	# and when output is 1, this throws off the neural network
	for i in range (0, train.__len__()):
		for j in range (1, train[i].__len__()):
			train[i][j] = train[i][j] / 1000
	
	#read the target labels
	target_labels = ri.read_labels(train_target_fn)
	target_digits = [[0.0 for x in range(0, 10)] for y in range (0,target_labels.__len__())]
	for i in range (target_labels.__len__()):
		target_digits[i] = digit_vector[target_labels[i]]
	
	learning_rate = 0.3
	momentum = 0.3
	n_in = train[0].__len__() - 1
	n_hidden = 300 		# for this dataset good results (less than 5% error) start at 300 hidden nodes and more
	n_out = 10
	
	dr_nn = nn.neural_network(learning_rate, momentum, n_in, n_hidden, n_out)
	
	total_iterations = 200
	
	for i in range (0, total_iterations):
		start = time.time()
		dr_nn.back_propagation_full_gradient_descent(train, target_digits)
		stop = time.time()
		
		print "back propagation for iteration {} took {} seconds.".format(i, stop - start)
		dr_nn.save_neural_network("tmp/nn" + "{}".format(i) + ".txt")
		
		
		err = 0.0
		
		for j in range(0, target_digits.__len__()):
			dr_out = dr_nn.compute_output(train[j])
			for k in range(0, dr_out.__len__()):
				err += (dr_out[k] - target_digits[j][k]) ** 2		# error is the euclidian distance
		
		print "iteration {} and the error is {}.".format(i, err)
	
	test_images = ri.read_images(test_fn)
	test_targets = ri.read_labels(test_target_fn)
	test_targets_digits = [[0.0 for x in range(0, 10)] for y in range (0,test_targets.__len__())]
	for i in range (test_targets_digits.__len__()):
		test_targets_digits[i] = digit_vector[test_targets[i]]
	
	correct_answers = 0
	
	start = time.time()
	for i in range (0, test_images.__len__()):
		out1 = dr_nn.compute_output(test_images[i])
		if vectors_identical(out1, test_targets_digits[i]):
			correct_answers += 1
		
	stop = time.time()
	
	print "testing {} items took {} seconds.".format(test_images.__len__(), stop - start)
	print "there are {} correct answers.".format(correct_answers)
		
	
	return dr_nn

def different_nns():
	train_fn = "train-images-idx3-ubyte"
	train_target_fn = "train-labels-idx1-ubyte"
	
	test_fn = "t10k-images-idx3-ubyte"
	test_target_fn = "t10k-labels-idx1-ubyte"
	
	train = ri.read_images(train_fn)
	target = ri.read_labels(train_target_fn)
	
	learning_rate = 0.3
	momentum = 0.3
	n_in = train[0].__len__()
	n_out = 1
	
	for i in range (5, 200):
		n_hidden = i
		
		start = time.time()
	
		dr_nn = nn.neural_network(learning_rate, momentum, n_in, n_hidden, n_out)
		dr_nn.back_propagation_full_gradient_descent(train, target)
		stop = time.time()
		print "training the neural network for {} hidden units took {} seconds.".format(n_hidden, stop - start)
		err = 0.0
		
		start = time.time()
	
		for j in range(0, target.__len__()):
			dr_out = dr_nn.compute_output(train[j])
			err += (dr_out[0] - target[j][0]) ** 2		# error is the euclidian distance
		
		stop = time.time()
		
		print "evaluating {} took {} seconds.".format(target.__len__(), stop - start)
		print "n_hidden {} and the error is {}.".format(i, err)
	
	return

def vectors_identical(x,y):
	if (x.__len__() != y.__len__()):
		return False
	
	max_x = -10000000.0
	max_y = -10000000.0
	max_x_index = -1
	max_y_index = -1
	
	for i in range (x.__len__()):
		if x[i] > max_x:
			max_x = x[i]
			max_x_index = i
		
		if y[i] > max_y:
			max_y = y[i]
			max_y_index = i
			
	
	return (max_x_index == max_y_index)