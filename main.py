import matplotlib.pyplot as plt

import bert
import Word2Vector
import tfid
import plot

epoch_n = 10
batch_size = 100


def main():
    print("IMDB Dataset Classification Testing")
    w2v_result = Word2Vector.word2Vec(epoch_n, batch_size)
    bert_result = bert.bert(epoch_n, batch_size)
    tfid_result = tfid.tfidCLS(epoch_n, batch_size)

    loss = [w2v_result[2], bert_result[2], tfid_result[2]]
    model_names = [w2v_result[0], bert_result[0], tfid_result[0]]

    plot.plot_points(loss, model_names)

    loss = [bert_result[2], tfid_result[2]]
    model_names = [bert_result[0], tfid_result[0]]

    plot.plot_points(loss, model_names)


if __name__ == "__main__":
    main()
