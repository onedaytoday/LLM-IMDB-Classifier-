import tokenizers
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import gensim.downloader as api
from torch import optim
from transformers import AutoModel, AutoTokenizer, Trainer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def Tfid():
    dataset = load_dataset("imdb")  # Loads dataset
    print(dataset)
    train_dataset = dataset['train']

    model = TfidfVectorizer()
    text_training_data_set = train_dataset["text"]

    print("Data Size is ", len(text_training_data_set))
    transform = model.fit(text_training_data_set)
    print("Feature Name", model.get_feature_names_out())
    print('--------------------------')
    print("Model Vocab size", len(model.vocabulary_))
    print('ID for ”ambitious"', model.vocabulary_.get('ambitious'))

    test_sentence = "I didn’t like the movie that much!"

    test_sentence_embedding = model.transform([test_sentence])
    arr = test_sentence_embedding.toarray()[0]
    print(len(arr))
    print(test_sentence_embedding.toarray())

    print("feature vector sum ", test_sentence_embedding.sum())

def transform(sentence):
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


def gensim():
    words = [
        "green", "red", "yellow",
        "farm", "sheep", "chicken",
        "vehicle", "motorcycle", "airplane"]

    model = api.load("glove-twitter-25")
    print(api.info("glove-twitter-25"))
    print(dir(model))
    print((model.vector_size))
    print(model.get_vector('this').sum())

    print('Similarity between', 'computer ', 'laptop ', model.similarity('computer', 'laptop'))
    print('Similarity between', 'computer ', 'fruit ', model.similarity('computer', 'fruit'))
    print('Similarity between', 'fruit ', 'banana ', model.similarity('fruit', 'banana'))
    print("-------------------------------")

    print('Distance between', 'france ', 'paris ', model.distance('france', 'paris'))
    print('Distance between', 'canada ', 'paris ', model.distance('canada', 'paris'))
    print('Distance between', 'brazil ', 'paris ', model.distance('brazil', 'paris'))

    similar_words = model.most_similar("boat", topn=10)

    print(similar_words)

    word_embeddings = []
    for i in words:
        word_embeddings.append(model.get_vector(i))
    print(word_embeddings)

    matrix = np.array(word_embeddings)
    myPCA = PCA(n_components=2)
    output = myPCA.fit_transform(matrix)
    plt.figure()
    plt.scatter(output[:, 0], output[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (output[i, 0], output[i, 1]))
    plt.show()


def bert():
    # Bert Uses a tokenizer to fully extract all the intricacies of the word such as hateful to 'hate' and 'ful'

    model_name = "distilbert-base-uncased"
    # https://huggingface.co/docs/transformers/en/model_doc/distilbert
    myModel = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('Vocab Size = ', tokenizer.vocab_size)
    print(tokenizer.tokenize('disrespectful'))
    print(tokenizer.tokenize('yes'))
    print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('hateful')))
    print("Natural Language Processing is fun! Tokenized is: ")
    print(tokenizer.tokenize('Natural language processing is fun!'))
    print(tokenizer.encode('Natural Language Processing is fun!'))
    print(myModel.config.max_position_embeddings)
    print('Hidden Size ', myModel.config.hidden_size)

Tfid()
gensim()
bert()
