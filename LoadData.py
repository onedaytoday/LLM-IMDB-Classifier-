from datasets import load_dataset


def load_data():
    # Sample corpus and labels
    dataset = load_dataset("imdb")  # Loads dataset
    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset = train_dataset.shuffle()
    test_dataset = test_dataset.shuffle()

    text_training_data_set = train_dataset["text"]
    training_label = train_dataset["label"]

    #text_training_data_set = text_training_data_set[:500]
    #training_label = training_label[:500]

    test_text = test_dataset["text"]
    test_label = test_dataset["label"]

    test_text = test_text[:200]
    test_label = test_label[:200]

    training_label = [int(x) for x in training_label]

    return text_training_data_set, training_label, test_text, test_label


def k_batch_data(training_features, training_labels, k):
    if k == 0 or k == 1:
        return [training_features], [training_labels]
    output_features = []
    output_labels = []
    if len(training_features) != len(training_labels):
        raise Exception()
    n = len(training_features)
    fold_size = n // k

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size
        output_features.append(training_features[start:end])
        output_labels.append(training_labels[start:end])
    return output_features, output_labels



load_data()
