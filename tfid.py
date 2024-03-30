import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import LoadData
import opt

number_of_output = 2


class TfidfClassifier(nn.Module):
    """
    Implement your TF-IDF based classifier here.
    """

    def __init__(self, vectorizer,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_dim=2):
        super().__init__()
        self.device = device

        self.vectorizer = vectorizer
        self.output_dim = output_dim
        self.layer1 = nn.Linear(len(self.vectorizer.vocabulary_), self.output_dim, device=self.device)

    def forward(self, x):
        input_layer_output = self.transform_data(x)
        input_layer_output_tensor = torch.tensor(input_layer_output,
                                                 device=self.device, dtype=torch.float32)
        output = self.layer1(input_layer_output_tensor)
        return output

    def transform_data(self, corpus):
        return self.vectorizer.transform(corpus).toarray()


def tfidCLS(epoch_n, batch_size):
    model = 'tfid'
    print('Model: ', model)

    text_training_data_set, training_label, text_test_data_set, test_label = LoadData.load_data()

    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the corpus
    tfidf_vectorizer.fit(text_training_data_set)

    # Initialize TfidfClassifier
    classifier = TfidfClassifier(vectorizer=tfidf_vectorizer, output_dim=2)

    # input to tensors
    training_label_stack = np.stack((training_label, [0 if x == 1 else 0 for x in training_label]), axis=1)

    # training_label_tensor = torch.tensor(training_label_stack, dtype=torch.float32, device=classifier.device)
    training_label_tensor = torch.tensor(training_label, dtype=torch.long, device=classifier.device)

    loss = opt.optimize(classifier, text_training_data_set, training_label_tensor, epoch_n=epoch_n,
                        num_of_batches=batch_size)
    results = opt.test_model(text_test_data_set, test_label, classifier)

    TestSentence = "The Dark Knight was a masterpiece! The plot, cast, and everything were absolutely sick!"
    classified_sentence = torch.argmax(classifier([TestSentence]).cpu())
    print(f'Sentence: "{TestSentence}" is classified as class {classified_sentence}')

    return model, results, loss, classifier


if __name__ == "__main__":
    tfidCLS(1, 1)
