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

        for iteration in range(self.max_iterations):
            print("Starting iteration %d..." % iteration)
            for i in range(len(trainingData)):
                vectors = util.Counter()
                for label in self.legalLabels:
                    vectors[label] = self.weights[label] * trainingData[i]

                #trainingLabels[i] is the true label
                best_guess_label = vectors.argMax()
                if trainingLabels[i] != best_guess_label:
                    self.weights[trainingLabels[i]] += trainingData[i]
                    self.weights[best_guess_label] -= trainingData[i]


    def classify(self, data):
        guesses = []
        for dat in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * dat
            guesses.append(vectors.argMax())
        return guesses