from Service import NaiveBayesianClassifier

class VectorClass:
    def __init__(self, class_label, vector):
        self.class_label = class_label
        self.vectors = list()
        self.vectors.append(vector)

    def add_vector(self, new_vector):
        self.vectors.append(new_vector)


