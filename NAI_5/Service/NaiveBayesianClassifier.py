import math


def get_summarize_dataset(data_set):
    summaries = [(get_mean(column), get_standard_deviation(column), len(column)) for column in zip(*data_set)]
    return summaries


def get_mean(values):
    return sum(values) / float(len(values))


def get_standard_deviation(values):
    avg = get_mean(values)
    variance = sum([(x - avg) ** 2 for x in values]) / float(len(values) - 1)
    return math.sqrt(variance)


def order_summarized_dataset(key, dataset):
    summary = dict()
    summary[key] = get_summarize_dataset(dataset)
    return summary


def get_probability(x, mean, standard_deviation):
    exponent = math.exp(-((x - mean) ** 2 / (2 * standard_deviation ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * standard_deviation)) * exponent


def calculate_class_probability(summarized_dataset, test_data):
    total_rows = sum([summarized_dataset[label][0][2] for label in summarized_dataset])
    probabilities = dict()
    for class_value, class_summaries in summarized_dataset.items():
        probabilities[class_value] = summarized_dataset[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= get_probability(test_data[i], mean, stdev)
    return probabilities

def get_precision(confusion_matrix):
    guess_list = [0, 0, 0]
    to_return = list()
    counter = 0
    for key in confusion_matrix:
        total = 0
        guess_list = confusion_matrix.get(key)
        part = guess_list[counter]
        for value in guess_list:
            total += value
        precision = part/total
        counter +=1
        to_return.append(precision)
        print(key, "Precision: ", precision)
    return to_return

def get_recall(confusion_matrix):
    counter = 0
    to_return = list()
    for key1 in confusion_matrix:
        total = 0
        for key2 in confusion_matrix:
            guess_list = confusion_matrix.get(key2)
            total += guess_list[counter]
        guest_part = confusion_matrix.get(key1)[counter]
        recall = guest_part/total
        counter += 1
        to_return.append(recall)
        print(key1, "Recall: ", recall)
    return to_return
def get_fscore(precision_list, recall_list):
    key_list = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    for precision, recall, key in zip(precision_list, recall_list, key_list):
        f1_score = 2*(precision*recall)/(precision+recall)
        print(key, "F1_Score: ", f1_score)

