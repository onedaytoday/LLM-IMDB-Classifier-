import gensim.downloader as api
import torch
import numpy as np
import torch.nn as nn

import LoadData
import opt

number_of_output = 2


class W2VClassifier(nn.Module):
    """
    Implement your word2vec based classifier here.
    """

    def __init__(self, pretrained_id="glove-twitter-25",
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.output_dim = 2
        self.device = device
        self.vectorizer = api.load(pretrained_id)
        layer1_input_size = self.vectorizer.vector_size
        self.layer1 = nn.Linear(in_features=layer1_input_size, out_features=self.output_dim, device=self.device)

    def forward(self, x):
        if type(x) == str:
            vector_form = self.transform_each(x)
        else:
            vector_form = self.transform_data(x)
        tensor_form = torch.tensor(vector_form, device=self.device, dtype=torch.float32)
        linear_output = self.layer1(tensor_form)
        output = linear_output
        return output

    def transform_data(self, x):
        output = []
        for row in x:
            output.append(self.transform_each(row))
        return output

    def transform_each(self, x):
        words = self.senetence_parse(x)
        vector_form = np.zeros(self.layer1.in_features)
        for word in words:
            try:
                vector_form = np.add(vector_form, self.vectorizer.get_vector(word))
            except Exception:
                pass
        return vector_form.tolist()

    def senetence_parse(self, sentence):
        # Initialize an empty string to store the cleaned sentence
        cleaned_sentence = ""

        # Iterate over each character in the sentence
        for char in sentence:
            # Check if the character is an alphabetic character or whitespace
            if char.isalpha() or char.isspace():
                # If it is an alphabetic character or whitespace, add it to the cleaned sentence
                cleaned_sentence += char.lower()  # Convert alphabetic characters to lowercase

        # Split the cleaned sentence into words based on whitespace
        words = cleaned_sentence.split()
        return words


def word2Vec(epoch_n, batch_size):
    model = 'word2vec'
    print('Model: ', model)

    text_training_data_set, training_label, text_test_data_set, test_label = LoadData.load_data()

    classifier = W2VClassifier("glove-twitter-25")

    # input to tensors
    training_label_stack = np.stack((training_label, [0 if x == 1 else 0 for x in training_label]), axis=1)
    training_label_tensor = torch.tensor(training_label, dtype=torch.long, device=classifier.device)
    # training_label_tensor = torch.tensor(training_label_stack, dtype=torch.float32, device=classifier.device)
    loss = opt.optimize(classifier, text_training_data_set, training_label_tensor, epoch_n=epoch_n,
                        num_of_batches=batch_size)
    # optimize(classifier, text_training_data_set, training_label_tensor, epoch_n)
    results = opt.test_model(text_test_data_set, test_label, classifier)

    TestSentence = "The Dark Knight was a masterpiece! The plot, cast, and everything were absolutely sick!"
    classified_sentence = torch.argmax(classifier([TestSentence]).cpu())
    print(f'Sentence: "{TestSentence}" is classified as class {classified_sentence}')

    return model, results, loss, classifier


if __name__ == "__main__":
    word2Vec(1, 1)
