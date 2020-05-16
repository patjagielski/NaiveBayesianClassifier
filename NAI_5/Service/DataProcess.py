from Model.VectorClass import VectorClass
from Service import NaiveBayesianClassifier
import numpy as np
import csv


def read_csv(filepath):
    dictionary_class_label_object = dict()
    to_summarize = np.zeros((105, 4))
    counter = 0
    with open(filepath, newline='') as csv_file:
        dialect = csv.Sniffer().sniff(csv_file.read(1024))
        csv_file.seek(0)
        reader = csv.reader(csv_file, dialect)

        for row in reader:
            vector_floats = list()
            class_label = ""
            to_summarize[counter] = row[0:4]
            for item in row:
                try:
                    number = float(item)
                    vector_floats.append(number)
                except:
                    class_label = item

            if class_label not in dictionary_class_label_object.keys():
                dictionary_class_label_object[class_label] = VectorClass(class_label, vector_floats)
            else:
                dictionary_class_label_object.get(class_label).add_vector(vector_floats)
            counter += 1
    return dictionary_class_label_object

def get_list_of_labels(filepath):
    to_summarize = [""]*45
    counter = 0

    with open(filepath, newline='') as csv_file:
        dialect = csv.Sniffer().sniff(csv_file.read(1024))
        csv_file.seek(0)
        reader = csv.reader(csv_file, dialect)
        for row in reader:
            to_summarize[counter] = row[4]
            counter += 1
    return to_summarize

def get_summarized_vector(vector_dictionary):
    keys = vector_dictionary.keys()
    summarized_dataset = dict()

    for key, value in zip(keys, vector_dictionary.values()):
        temp_dicionary = NaiveBayesianClassifier.order_summarized_dataset(key, value.vectors)
        temp_list = list(temp_dicionary.values())
        for number in temp_list:
            summarized_dataset[list(temp_dicionary.keys())[0]] = number
    return summarized_dataset
