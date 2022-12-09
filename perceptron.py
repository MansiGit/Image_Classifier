import util
PRINT = True

class PerceptronClassifier:
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() 

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0].keys()
        for iteration in range(self.max_iterations):
            print ("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
              "*** YOUR CODE HERE ***" 
              y_value = self.classify([trainingData[i]])[0]


              if y_value != trainingLabels[i]:
                #weights vector adjustment
                  self.weights[trainingLabels[i]] += trainingData[i] # encourage the actual answer : add weight phi to weight vector 
                  self.weights[y_value] -= trainingData[i] #punish the incorrect guess's  weight vector

           
    def classify(self, data):
        guesses = []
        for d in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * d
            guesses.append(vectors.argMax())

        return guesses


    def findHighWeightFeatures(self, label):
        """
        We get a list of 100 features with the highest weight
        """
        
        list_of_weights = []

        weight_values = self.weights[label]

        for i in range(300):
            curr_wt = weight_values.argMax()
            list_of_weights.append(curr_wt)
            weight_values[curr_wt]=-9999999999

        return list_of_weights