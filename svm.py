import numpy as np
from scipy.io import loadmat
from time import time
from sklearn.svm import SVC
import matplotlib.pyplot as plt

"""
MNIST data preprocess

"""
def preprocess():

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.01):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label

"""
Support Vector Machine

"""

print('\n\n--------------SVM-------------------\n\n')

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# Reshaping input for labels
train_label = np.squeeze(train_label)
validation_label = np.squeeze(validation_label)
test_label = np.squeeze(test_label)

# Linear kernel
clf = SVC(kernel='linear')

start_time = time()

clf.fit(train_data, train_label)

end_time = time()

print("\n\n----------------SVM with linear kernel---------------\n\n")
print("\n Time used: " + str(int(end_time - start_time)) + " seconds")
print("\n Accuracy on Training dataset: " + str(round(clf.score(train_data, train_label)*100, 2)) + "%")
print("\n Accuracy on Validation dataset: " + str(round(clf.score(validation_data, validation_label)*100, 2)) + "%")
print("\n Accuracy on Testing dataset: " + str(round(clf.score(test_data, test_label)*100, 2)) + "%")


# Radial kernel, gamma=1
clf = SVC(kernel='rbf', gamma=1)

start_time = time()

clf.fit(train_data, train_label)

end_time = time()

print("\n\n-----------SVM with radial kernel, gamma=1-----------\n\n")
print("\n Time used: " + str(int(end_time - start_time)) + " seconds")
print("\n Accuracy on Training dataset: " + str(round(clf.score(train_data, train_label)*100, 2)) + "%")
print("\n Accuracy on Validation dataset: " + str(round(clf.score(validation_data, validation_label)*100, 2)) + "%")
print("\n Accuracy on Testing dataset: " + str(round(clf.score(test_data, test_label)*100, 2)) + "%")


# Radial kernel, gamma=default
clf = SVC(kernel='rbf', gamma='auto')

start_time = time()

clf.fit(train_data, train_label)

end_time = time()

print("\n\n----------SVM with radial kernel, gamma=auto---------\n\n")
print("\n Time used: " + str(int(end_time - start_time)) + " seconds")
print("\n Accuracy on Training dataset: " + str(round(clf.score(train_data, train_label)*100, 2)) + "%")
print("\n Accuracy on Validation dataset: " + str(round(clf.score(validation_data, validation_label)*100, 2)) + "%")
print("\n Accuracy on Testing dataset: " + str(round(clf.score(test_data, test_label)*100, 2)) + "%")


# Radial kernel, C=(1,10,20,30,··· ,100)
a = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
train_acc = []
validation_acc = []
test_acc = []

for i in a:
    
    clf = SVC(kernel='rbf', C=i)

    start_time = time()
    
    clf.fit(train_data, train_label)
    
    end_time = time()
    
    train_accuracy = clf.score(train_data, train_label)*100
    validation_accuracy = clf.score(validation_data, validation_label)*100
    test_accuracy = clf.score(test_data, test_label)*100
    
    train_acc.append(train_accuracy)
    validation_acc.append(validation_accuracy)
    test_acc.append(test_accuracy)
    
    print("\n\n----------SVM with radial kernel, C=" + i + "---------\n\n")
    print("\n Time used: " + str(int(end_time - start_time)) + " seconds")
    print("\n Accuracy on Training dataset: " + str(round(clf.score(train_data, train_label)*100, 2)) + "%")
    print("\n Accuracy on Validation dataset: " + str(round(clf.score(validation_data, validation_label)*100, 2)) + "%")
    print("\n Accuracy on Testing dataset: " + str(round(clf.score(test_data, test_label)*100, 2)) + "%")

plt.plot(a, train_acc, marker='.')
plt.plot(a, validation_acc, marker='.')
plt.plot(a, test_acc, marker='.')
plt.legend(['Training', 'Validation', 'Test'])
plt.xlabel('Value of C')
plt.ylabel('Accuracy, %')

plt.show()