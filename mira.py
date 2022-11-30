# # Mira implementation
# import util
# PRINT = True

# class MiraClassifier:
#   """
#   Mira classifier.
  
#   Note that the variable 'datum' in this code refers to a counter of features
#   (not to a raw samples.Datum).
#   """
#   def __init__( self, legalLabels, max_iterations):
#     self.legalLabels = legalLabels
#     self.type = "mira"
#     self.automaticTuning = False 
#     self.C = 0.001
#     self.legalLabels = legalLabels
#     self.max_iterations = max_iterations
#     self.initializeWeightsToZero()

#   def initializeWeightsToZero(self):
#     "Resets the weights of each label to zero vectors" 
#     self.weights = {}
#     for label in self.legalLabels:
#       self.weights[label] = util.Counter() # this is the data-structure you should use
  
#   def train(self, trainingData, trainingLabels, validationData, validationLabels):
#     "Outside shell to call your method. Do not modify this method."  
      
#     self.features = list(trainingData[0].keys()) # this could be useful for your code later...
    
#     if (self.automaticTuning):
#         Cgrid = [0.002, 0.004, 0.008]
#     else:
#         Cgrid = [self.C]
        
#     return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

#   def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
#     """
#     This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
#     then store the weights that give the best accuracy on the validationData.
    
#     Use the provided self.weights[label] data structure so that 
#     the classify method works correctly. Also, recall that a
#     datum is a counter from features to values for those features
#     representing a vector of values.
#     """
#     "*** YOUR CODE HERE ***"
#     util.raiseNotDefined()

#   def classify(self, data ):
#     """
#     Classifies each datum as the label that most closely matches the prototype vector
#     for that label.  See the project description for details.
    
#     Recall that a datum is a util.counter... 
#     """
#     guesses = []
#     for datum in data:
#       vectors = util.Counter()
#       for l in self.legalLabels:
#         vectors[l] = self.weights[l] * datum
#       guesses.append(vectors.argMax())
#     return guesses

  
#   def findHighOddsFeatures(self, label1, label2):
#     """
#     Returns a list of the 100 features with the greatest difference in feature values
#                      w_label1 - w_label2

#     """
#     featuresOdds = []

#     "*** YOUR CODE HERE ***"

#     return featuresOdds



# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        greatest_w = dict()
        greatest_acc = float('-inf')

        for c in Cgrid:
            current_weight = self.weights.copy()
            for n in range(self.max_iterations):

                for x, w in enumerate(trainingData):

                    predict_score = float('-inf')
                    predict_label = float('-inf')
                    for label in self.legalLabels:
                        if w * current_weight[label] > predict_score:
                            predict_score, predict_label = w * current_weight[label], label

                    actual_label = trainingLabels[x]
                    if predict_label != actual_label:
                        f_val = w.copy()
                        minimized_tau = min(c, ((current_weight[predict_label] - current_weight[actual_label]) * f_val + 1.0) / (2.0 * (f_val * f_val)))
                        f_val.divideAll(1.0 / minimized_tau)
                        current_weight[predict_label] = current_weight[predict_label] - f_val
                        current_weight[actual_label] = current_weight[actual_label] + f_val

            guesses = self.classify(validationData)
            correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
            accuracy = float(float(correct) / len(guesses))

            if accuracy > greatest_acc:
                greatest_acc = accuracy
                greatest_w = current_weight

        self.weights = greatest_w

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
                         w_label1 - w_label2

        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"

        return featuresOdds


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
PRINT = True

class kNearestNeighborsClassifier:
  """
  Perceptron classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, k=10):
    self.legalLabels = legalLabels
    self.type = "question3"
    self.k = k
    self.weights = {}
  #   for label in legalLabels:
  #     self.weights[label] = util.Counter() # this is the data-structure you should use

  # def setWeights(self, weights):
  #   assert len(weights) == len(self.legalLabels);
  #   self.weights == weights;
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    self.trainingData = self.downscaleDataFunction(trainingData)
    self.trainingLabels = trainingLabels
    


  def downscaleDataFunction(self, datum_list):
    DATA_HEIGHT, DATA_WIDTH = 0,0
    BLOCK_HEIGHT, BLOCK_WIDTH = 0,0
    BLOCK_ROWS, BLOCK_COLS = 0,0
    if 2 in self.legalLabels:
      DATA_HEIGHT, DATA_WIDTH = 28,28
      BLOCK_HEIGHT, BLOCK_WIDTH = 4,4
      BLOCK_ROWS, BLOCK_COLS = 7,7

    else:
      DATA_HEIGHT, DATA_WIDTH = 70,60
      BLOCK_HEIGHT, BLOCK_WIDTH = 7,6
      BLOCK_ROWS, BLOCK_COLS = 10,10

    downscaledDataAll = []
    for data in datum_list:
      downscaledData = util.Counter()
      for i_big in range(BLOCK_ROWS):
        for j_big in range(BLOCK_COLS):
          isFeature = 0

          for i_small in range(BLOCK_HEIGHT):
            if isFeature:
              break
            for j_small in range(BLOCK_WIDTH):
              if data[( i_big*BLOCK_HEIGHT + i_small , j_big*BLOCK_WIDTH + j_small )] == 1:
                isFeature = 1
                break

          downscaledData[(i_big,j_big)] = isFeature

      downscaledDataAll.append(downscaledData)

    return downscaledDataAll

  def findDistance(self, test_datum, train_data):
    if True:
      x = test_datum - train_data
      return numpy.sum(numpy.abs([x[value] for value in x]))
    
  def classify(self, data ):
    """
    Find the k closest 'neighbors' of the test image in the training data
    and then return the label which appeared the most. If there is a tie
    then pick the label of the training image with the lowest distance.
    """

    data = self.downscaleDataFunction(data)

    guesses = []
    for datum in data:
      distanceValues = []
      # print("------")
      # start = time.time()
      for i in range(len(self.trainingData)):
        distanceValues.append(  (self.findDistance(datum,self.trainingData[i]), i)  ) # need to pass i through for each distance to get the trainingLabel
      # end1 = time.time() - start
      # print(end1)

      distanceValues.sort()
      distanceValues = distanceValues[:self.k]

      bestK_labels = []
      for distance in distanceValues:
        bestK_labels.append(self.trainingLabels[distance[1]])

      try:
        guesses.append(statistics.mode(bestK_labels))
      except:
        guesses.append(bestK_labels[0])


    return guesses

