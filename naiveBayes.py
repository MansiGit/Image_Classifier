import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 0.001 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.dict_phi_value_per_feature= None

  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = list(trainingData[0].keys()) # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    frequency = util.Counter() #initialize a dict that counts the freq of each digit or each frew of T or false for faces
    for i in trainingLabels: frequency[i]+=1

    for i in frequency.keys():
      frequency[i]=frequency[i]/len(trainingData) #for face data {0: 0.51483894234 1: 49.324892893643}
    
    dict_trainingdata_by_label={key: list() for key in trainingLabels}
    
    #segregate training data by label 
    for i in range(len(trainingData)):
      dict_trainingdata_by_label[trainingLabels[i]].append(trainingData[i])
   
    ###############################
    # Finding phi values for each pixel for each label
    ###############################
    dict_phi_value_per_feature={key: util.Counter() for key in trainingLabels}
    
    for label in dict_trainingdata_by_label.keys():
      temp=util.Counter()
      cnt=0
      for d in dict_trainingdata_by_label[label]:
        cnt+=1
        for key,value in d:
          if(value>0):
            temp[key]+=1  
      for loc,freq in temp:
        if(freq==0):
          freq=self.k
        temp[loc]=float(freq/cnt)
      dict_phi_value_per_feature.append(temp)

    self.dict_phi_value_per_feature=dict_phi_value_per_feature
    #"*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    logJoint = util.Counter()
    
    #"*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    """
    featuresOdds = []
        
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds
    

    
      
