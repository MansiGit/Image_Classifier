# KNN start

# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation
import util
import numpy
import math
import statistics
import tracemalloc
import time
import tensorflow as tf
from tensorflow.keras import layers
PRINT = True
matrix = []

class kNearestNeighborsClassifier:

	def __init__( self, legalLabels, k=10):
		self.legalLabels = legalLabels
		self.type = "question3"
		self.k = k
		self.weights = {}
	  	
	
	def train( self, trainingData, trainingLabels, validationData, validationLabels ):
		print("training data")
		
		matrixA = []
		matrixB = []

		for dicti in trainingData:
			matrixA.append(dicti.values())

		
		for dictiLabels in trainingLabels:
			matrixB.append(dictiLabels.values())
	
		model = tf.keras.Sequential([layers.Conv2D(filters=10, kernel_size=3, activation="relu", input_shape=(28,  28,  1)),layers.Conv2D(10,  3, activation="relu"), layers.MaxPool2D(), layers.Conv2D(10,  3, activation="relu"), layers.Conv2D(10,  3, activation="relu"), layers.MaxPool2D(), layers.Flatten(), layers.Dense(10, activation="softmax")])
		model.summary()
		model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
		model.fit(matrixA, matrixB, epochs=10)

		
	def classify(self, testData):
		model.fit(testData, testLabel, epochs=10)
		model.evaluate(testData, testLabel)

