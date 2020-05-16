from Service import DataProcess
from Service import NaiveBayesianClassifier
from Service import CreateConfusionMatrix
import operator


reading_test_file = DataProcess.read_csv(r"C:\Users\jagie\PycharmProjects\NAI_Projects\NAI_5\Resources\iris-test.data")
reading_training_file = DataProcess.read_csv(r"C:\Users\jagie\PycharmProjects\NAI_Projects\NAI_5\Resources\iris.data")

test_values = list(reading_test_file.values())
test_keys = DataProcess.get_list_of_labels(r"C:\Users\jagie\PycharmProjects\NAI_Projects\NAI_5\Resources\iris-test.data")


summary = DataProcess.get_summarized_vector(reading_training_file)
confusion_matrix = CreateConfusionMatrix.create_confusion_matrix(test_values, test_keys, summary)
print("\nPrecision:")
precision_list = NaiveBayesianClassifier.get_precision(confusion_matrix)
print("\nRecall:")
recall_list = NaiveBayesianClassifier.get_recall(confusion_matrix)
print("\nF-Score:")
NaiveBayesianClassifier.get_fscore(precision_list, recall_list)

user_input = input("Would you like to continue? [yes/no]")
while "yes" in user_input:
    a = float(input("a: "))
    b = float(input("b: "))
    c = float(input("c: "))
    d = float(input("d: "))
    user_list = [a, b, c, d]
    guess = NaiveBayesianClassifier.calculate_class_probability(summary, user_list)
    print(max(guess.items(), key=operator.itemgetter(1))[0])

    user_input = input("Would you like to continue? [yes/no]")
    if "yes" in user_input:
        continue
    else:
        exit()




