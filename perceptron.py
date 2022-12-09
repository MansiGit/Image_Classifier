import util

class PerceptronClassifier:
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() 

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0].keys()
        for iteration in range(self.max_iterations):
            print ("Starting iteration ", iteration, "...")
            for data_index in range(len(trainingData)): 
              val_y = self.classify([trainingData[data_index]])[0]
              if val_y != trainingLabels[data_index]:
                  self.weights[trainingLabels[data_index]] = self.weights[trainingLabels[data_index]] + trainingData[data_index]  
                  self.weights[val_y] = self.weights[val_y] - trainingData[data_index] 

           
    def classify(self, data):
        guesses = []
        for dat in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * dat
            guesses.append(vectors.argMax())
        return guesses