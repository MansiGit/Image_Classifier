# KNN start

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
import math
import ast
import numpy as np
import random

import operator 
from operator import itemgetter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

class kNearestNeighborsClassifier:

    # def __init__( self, legalLabels, k=10):
    #     self.legalLabels = legalLabels
    #     self.type = "knnClassifier"
    #     self.k = k
    #     self.weights = {}
          
    def preprocessData(self, trainingNumber):
        

        txt_file = open("KNN_DATA/traindata_digits_preprocessed.txt", "r")
        new_file_content = txt_file.read()
        main_list = ast.literal_eval(new_file_content)
        for i in range(len(main_list)):
            main_list[i] = np.array(main_list[i])

        label_txt_file = open("KNN_DATA/traininglabels", "r")
        label_file_content = label_txt_file.read()
        label_file_content = label_file_content.split('\n')
        #label_file_content = ast.literal_eval(label_file_content)

        #file_content = [n for n in file_content]
        #file_content

        label_list = ' '.join(label_file_content).split()
        for i in range(len(label_list)):
            label_list[i]=int(label_list[i])

        merged = list(map(lambda x, y:(x,y), main_list, label_list))
        random.shuffle(merged)
        main_list,label_list=[],[]
        for i in range(trainingNumber):
            main_list.append(merged[i][0])
            label_list.append(merged[i][1])
        

        # TESTDATA
        txt_file = open("KNN_DATA/testdata_digits_preprocessed.txt", "r")
        new_file_content = txt_file.read()
        test_main_list = ast.literal_eval(new_file_content)

        for i in range(len(test_main_list)):
            test_main_list[i] = np.array(test_main_list[i])


        test_label_txt_file = open("KNN_DATA/testlabels", "r")
        test_label_file_content = test_label_txt_file.read()
        test_label_file_content=test_label_file_content.split('\n')


        test_label_list=' '.join(test_label_file_content).split()
        for i in range(len(test_label_list)):
            test_label_list[i]=int(test_label_list[i])

        x=0
        some_digit = test_main_list[x]
        some_digit_image = some_digit.reshape(28, 28)
        # uncomment later
        # plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
        #plt.axis(“off”)

        # uncomment later
        # plt.show()
        print(" ")
        print("Actual Value: "+str(test_label_list[x]))


        X = main_list
        y = label_list[:5000]


        X_train = X
        y_train = y
        X_test = test_main_list
        y_test = test_label_list[:100]


        kVals = np.arange(3,9,2)
        accuracies = []
        for k in kVals:
            model = KNN(K = k)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            accuracies.append(acc)
            print("K = "+str(k)+"; Accuracy: "+str(acc))

        max_index = accuracies.index(max(accuracies))
        print(max_index)

        model = KNN(K = kVals[max_index])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print("K = "+str(max_index*2+3)+"; Accuracy: "+str(acc))

        x=67
        some_digit = test_main_list[x]
        some_digit_image = some_digit.reshape(28, 28)
        # uncomment later
        # plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
        #plt.axis(“off”)

        # uncomment later
        # plt.show()
        print(" ")
        print("Actual Value: "+str(test_label_list[x]))
        print("Predicted Value: "+str(pred[x]))


        plt.plot(kVals, accuracies) 
        plt.xlabel("K Value") 
        plt.ylabel("Accuracy")

        return acc, 0

class KNN:
    def __init__(self, K=3):
        self.K = K
    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
    def euc_dist(x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    def predict(self, X_test):
      predictions = [] 
      for i in range(len(X_test)):
          dist = np.array([KNN.euc_dist(X_test[i], x_t) for x_t in self.X_train])
          dist_sorted = dist.argsort()[:self.K]
          neigh_count = {}
          for idx in dist_sorted:
              if self.Y_train[idx] in neigh_count:
                  neigh_count[self.Y_train[idx]] += 1
              else:
                  neigh_count[self.Y_train[idx]] = 1
          sorted_neigh_count = sorted(neigh_count.items(),    
          key=operator.itemgetter(1), reverse=True)
          predictions.append(sorted_neigh_count[0][0]) 
      return predictions