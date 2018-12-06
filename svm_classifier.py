import mnist_loader
from sklearn import svm


def svm_baseline():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # Fit
    classifier = svm.SVC()
    classifier.fit(training_data[0], training_data[1])
    # Predict
    predictions = [int(a) for a in classifier.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print("Baseline classifier using an SVM.")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))


if __name__ == "__main__":
    svm_baseline()
