from transformers import AutoModel, AutoTokenizer
import LoadData
import torch
import numpy as np
import torch.nn as nn

import opt

number_of_output = 2



class BERTClassifier(nn.Module):

    def __init__(self, model_name="distilbert-base-uncased", output_dim=2,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.myModel = AutoModel.from_pretrained(model_name)
        self.myModel.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.set_requires_grad_to_false(self.myModel)
        self.layer1 = nn.Linear(in_features=self.myModel.config.hidden_size,
                                out_features=self.output_dim, device=self.device)

    def forward(self, x):
        input_layer_output = self.transform_data(x)
        input_ids = torch.tensor(input_layer_output, dtype=torch.long, device=self.device)
        model_output = self.myModel(input_ids)
        model_output_to_tensor = model_output.last_hidden_state[:, 0, :]
        output = self.layer1(model_output_to_tensor)
        return output

    def transform_data(self, corpus):
        output = []
        for i in corpus:
            tokenized_sentence = self.tokenizer.encode(i,
                                                       max_length=self.myModel.config.max_position_embeddings,
                                                       truncation=True,
                                                       padding='max_length'
                                                       )
            output.append(tokenized_sentence)
        return output

    def set_requires_grad_to_false(self, model):
        for param in model.parameters():
            param.requires_grad = False


def bert(epoch_n, batch_size):
    model = 'bert'
    print('Model: ', model)
    text_training_data_set, training_label, text_test_data_set, test_label = LoadData.load_data()

    # Initialize TfidfClassifier
    classifier = BERTClassifier()

    # input to tensors
    training_label_stack = np.stack((training_label, [0 if x == 1 else 0 for x in training_label]), axis=1)

    # training_label_tensor = torch.tensor(training_label_stack, dtype=torch.float32, device=classifier.device)
    training_label_tensor = torch.tensor(training_label, dtype=torch.long, device=classifier.device)

    loss = opt.optimize(classifier, text_training_data_set,
                        training_label_tensor, epoch_n=epoch_n, num_of_batches=batch_size)
    results = opt.test_model(text_test_data_set, test_label, classifier)

    TestSentence = "The Dark Knight was a masterpiece! The plot, cast, and everything were absolutely sick!"
    classified_sentence = torch.argmax(classifier([TestSentence]).cpu())
    print(f'Sentence: "{TestSentence}" is classified as class {classified_sentence}')

    return model, results, loss, classifier


if __name__ == "__main__":
    bert(1, 1)
# 0.75
# after 10 epoch of 100 batches
