import os
import cv2
import pickle
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from argparse import ArgumentParser

DATA_EXTRACTED_FILE_PATH = os.path.join("data.pickle")

def extract_features(file_path, size):
    data = []
    file = open(file_path, "r")
    for file_line in file:
        line_splitted = file_line.split(" ")
        data_path = os.path.join(line_splitted[0])
        label = line_splitted[1].replace("\n", '')
        image = cv2.imread(data_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

        pixel_array = []
        for i in image: 
            for j in i:
                pixel_array.append(j)

        data.append([pixel_array, int(label)])
        
    pick_in = open(DATA_EXTRACTED_FILE_PATH, 'wb')
    pickle.dump(data, pick_in)
    pick_in.close()


def train(test_size, neighbors, metric):
    pick_in = open(DATA_EXTRACTED_FILE_PATH, 'rb')
    data = pickle.load(pick_in)
    pick_in.close()

    features = []
    labels = []

    for feature, label in data: 
        features.append(feature)
        labels.append(label)

    X_train, X_test, y_train, y_test =  train_test_split(features, labels, test_size = test_size)

    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    neigh = KNeighborsClassifier(n_neighbors = neighbors, metric = metric)

    print ('Fitting knn')
    neigh.fit(X_train, y_train)

    print ('Predicting...')
    y_pred = neigh.predict(X_test)

    print ('Accuracy: ',  neigh.score(X_test, y_test))

    cm = confusion_matrix(y_test, y_pred)
    print (cm)
    print(classification_report(y_test, y_pred))

argument_parser = ArgumentParser()
argument_parser.add_argument("-d", "--description", required = True, help = "Path to txt file. The txt file describes, per line, the path to image and its label. Example: data/image.jpg 0")
argument_parser.add_argument("-ts", "--testsize", required = False, default = 0.5, type = float, help = "Percent of the dataset used to test")
argument_parser.add_argument("-n", "--neighbors", required = False, default = 3, type = int, help = "How many neighbors to KNN classifier")
argument_parser.add_argument("-s", "--size", required = False, default = 30, type = int, help = "Size to resize images")
argument_parser.add_argument("-m", "--metric", required = False, default = "euclidean", help = "Metric to KNN classifier")

arguments = vars(argument_parser.parse_args())

description = arguments["description"]
file_exists = os.path.exists(description) and os.path.isfile(description)
if not file_exists: print("The file does not exists or is not a file") and exit() 

test_size = arguments["testsize"]
if test_size < 0.0 or test_size > 1.0: print("Invalid percent") and exit()

neighbors = arguments["neighbors"]
if neighbors <= 0: print("Invalid neighbors for KNN") and exit()

size = arguments["size"]
if size <= 0: print("Invalid size for resizing") and exit()

metric = arguments["metric"]
print("I' assuming that you are selecting a valida metric") #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

extract_features(file_path = description, size = size)
train(test_size = test_size, neighbors = neighbors, metric = metric)
