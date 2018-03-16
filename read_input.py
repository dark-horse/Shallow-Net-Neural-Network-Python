import sys
import os
import struct
import numpy as np

def read_images(fn):
	"""
	This method reads the data in the MNIST data base of handwritten digits.
	The method returns an image vector to be used in a neural network
	The method does not arrange the image vector into columns / rows
	"""
	f = open(fn, "rb")			#"rb" because we are opening a binary file
	
	#read the "MAGIC" (????) number
	tmp = f.read(4)
	i = struct.unpack(">i", tmp)		# ">i" because the file is in big endian mode
	if i[0] != 2051:
		raise ValueError ("wrong file format")
	
	#read the number of images
	tmp = f.read(4)
	i = struct.unpack(">i", tmp)		# ">i" because the file is in big endian mode
	img_count = i[0]
	
	#read the number of rows in an image
	tmp = f.read(4)
	i = struct.unpack(">i", tmp)		# ">i" because the file is in big endian mode
	row_count = i[0]
	
	#read the number of columns in an image
	tmp = f.read(4)
	i = struct.unpack(">i", tmp)		# ">i" because the file is in big endian mode
	col_count = i[0]
	
	# each image consists of exactly col_count x row_count pixels.
	# each pixel is exactly 1 byte.
	
	img_vector = np.empty((img_count, col_count * row_count+1), dtype=np.float64)
	for i in range (0, img_count):
		img_vector[i,0] = 1.0
		for j in range (0, col_count * row_count):
			tmp = f.read(1)
			img_vector[i,j+1] = float(struct.unpack("B", tmp)[0])

	f.close()
	return img_vector
	
	
def read_labels(fn):
	"""
	This method reads the labels associated with the data in the MNIST data base of handwritten digits
	This method returns a "target" vector to be used in a neural network
	"""
	
	f = open(fn, "rb")			#"rb" because we are opening a binary file

	#read the "MAGIC" (????) number
	tmp = f.read(4)
	i = struct.unpack(">i", tmp)		# ">i" because the file is in big endian mode
	if i[0] != 2049:
		raise ValueError ("wrong file format")
	
	#read the number of labels
	tmp = f.read(4)
	i = struct.unpack(">i", tmp)		# ">i" because the file is in big endian mode
	lbl_count = i[0]
	
	lbl_vector = [0 for j in range(0, lbl_count)]
	
	for i in range (0, lbl_count):
		tmp = f.read(1)
		lbl_vector[i] = struct.unpack("B", tmp)[0]
		
	f.close()
	return lbl_vector
	
	
