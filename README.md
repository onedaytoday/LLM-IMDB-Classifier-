Project 2 - IMDB Review Classification

Objectives:

- Task 1: Explore TfidfVectorizer
- Task 2: Explore Word2Vector Model glove-twitter-25
- Task 3: Explore distilbert-base-uncased
- Task 4: Compare the model’s classification abilities using IMDB dataset

Data:

- IMDB: https://huggingface.co/datasets/imdb
- Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.

Implementation: 

Libraries

- Torch - neural network optimization
- TfidfVectorizer- used for data parsing and processing
- gensim - Word2Vec
- Transformers - used to load pretrained models 

Hyperparameters 

- Batch Size = 100 batches (250 entries per batch) 
- Epoch = 10
- Test Size= 200 randomize entries 
- Learning Rate= 0.01 

Algorithm and Code 

The comparison code is divided into three main python files each containing the classifications model used for testing. In addition, there are additional utilizing functions for testing the models used across all of the testing. 

Results: 

Summary:



|Model|Accuracy|Recall|F1|
| - | - | - | - |
|Word2Vec|0\.56|0\.0112|0\.022|
|Bert|0\.81|0\.78|0\.804|
|Tfid|0\.735|0\.72|0\.746|

Task 1 Results:

TfidfVectorizer

Vocab Size = 74849

ID for ”ambitious" =  2977

feature vector sum =  2.294023319093611

Task 2 Results:

Word2Vector

Similarity between computer and  laptop  = 0.8352675 Similarity between computer and fruit =  0.45673344 Similarity between fruit  and banana =  0.8357839 -------------------------------

Distance between france and paris  0.11305677890777588 Distance between canada and paris  0.28165972232818604 Distance between brazil and  paris  0.3328576683998108

Closest words to ”boat” = ['cabin', 'truck', 'pool', 'plane', 'flying', 'balloon', 'roof', 'rides', 'backyard', 'cab']

![](Pic1.png)

Task 3 Results:

Bert

Vocab Size =  30522 Hidden Size = 768

Bert uses a tokenizer to preprocess text inputs into a format that can be effectively processed by the mode. This allows the model to go beyond the syntax of the language and further break down meaning by breaking down words and its meaning. It also allows for a more predictable input handling.

['natural', 'language', 'processing', 'is', 'fun', '!'] [101, 3019, 2653, 6364, 2003, 4569, 999, 102]

What is input ID: Input IDs are numerical representations of tokenized input sequences in BERT, mapping each token to a unique integer ID through a pre-trained tokenizer. They form the basis of the input data, converting text into a format suitable for processing by the model.

Why Do we use an attention Mask? The attention mask in BERT is a binary tensor indicating which tokens should be attended to (with a value of 1) and which ones should be ignored (with a value of 0). 

Task 4 Results:

Model Classification for “The Dark Knight was a masterpiece! The plot, cast, and everything were absolutely sick!”

Bert = 1 Word2Vec = 1 Tfid = 1

![](Pic2.png)

![](Pic3.png)
