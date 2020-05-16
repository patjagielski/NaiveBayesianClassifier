from Service import NaiveBayesianClassifier
import operator


def create_confusion_matrix(test_values, test_keys, summary):
    counter = -1
    correct_guess = 0
    incorrect_guess = 0
    total_correct = 0
    total= 0
    setosa = [0, 0, 0]
    versicolor = [0, 0, 0]
    virginica = [0, 0, 0]
    confusion_matrix = {"Iris-setosa": setosa,
                        "Iris-versicolor": versicolor,
                        "Iris-virginica": virginica}
    for vector_list in test_values:
        correct_guess = 0
        incorrect_guess = 0
        for vector in vector_list.vectors:
            bayesian_classifications = NaiveBayesianClassifier.calculate_class_probability(summary, vector)
            guess = max(bayesian_classifications.items(), key=operator.itemgetter(1))[0]
            counter += 1
            actual = test_keys[counter]
            # print("Guess",guess)
            # print("Actual",actual)
            if guess in actual:
                correct_guess += 1
                if guess in "Iris-setosa":
                    setosa[0] = correct_guess
                    confusion_matrix[guess] = setosa
                elif guess in "Iris-versicolor":
                    versicolor[1] = correct_guess
                    confusion_matrix[guess] = versicolor
                elif guess in "Iris-virginica":
                    virginica[2] = correct_guess
                    confusion_matrix[guess] = virginica
            else:
                incorrect_guess += 1
                if actual in "Iris-setosa":
                    if guess in "Iris-versicolor":
                        virginica[1] = incorrect_guess
                        confusion_matrix[actual] = virginica
                    elif guess in "Iris-virginica":
                        virginica[2] = incorrect_guess
                        confusion_matrix[actual] = virginica
                elif actual in "Iris-versicolor":
                    if guess in "Iris-virginica":
                        virginica[2] = incorrect_guess
                        confusion_matrix[actual] = virginica
                    elif guess in "Iris-setosa":
                        virginica[0] = incorrect_guess
                        confusion_matrix[actual] = virginica
                elif actual in "Iris-virginica":
                    if guess in "Iris-versicolor":
                        virginica[1] = incorrect_guess
                        confusion_matrix[actual] = virginica
                    elif guess in "Iris-setosa":
                        virginica[0] = incorrect_guess
                        confusion_matrix[actual] = virginica
            total_correct += correct_guess
            total += correct_guess+incorrect_guess
    print(list(confusion_matrix.keys())[0], "   ", list(confusion_matrix.values())[0])
    print(list(confusion_matrix.keys())[1], list(confusion_matrix.values())[1])
    print(list(confusion_matrix.keys())[2], "", list(confusion_matrix.values())[2])
    print("Accuracy: ", format(total_correct / total, '%'))
    return confusion_matrix
