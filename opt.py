import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import optim

import LoadData

learning_rate = .01

def optimize(model, x, y, epoch_n, num_of_batches=1):
    # Loss function and optimizer
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    x_batches, y_batches = LoadData.k_batch_data(x, y, num_of_batches)
    output = []
    # Training loop
    for epoch in range(epoch_n):
        # Print loss every 5 epochs
        for batch in range(num_of_batches):
            x_batch = x_batches[batch]
            y_batch = y_batches[batch]
            loss = each_batch(model, criterion, optimizer, x_batch, y_batch)
            output.append(loss.item())
            if batch % 50 == 0 or batch+1 == num_of_batches:
                print(f'Epoch [{epoch + 1}/{epoch_n}] '
                      f'[Batch {batch+1}/{num_of_batches}], Loss: {loss.item():.4f}')
    return output


def each_batch(model, criterion, optimizer, x, y):
    # Forward pass: Compute predicted y by passing x to the model
    pred_y = model(x)

    # Compute loss
    loss = criterion(pred_y, y)

    # Zero gradients, backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def test_model(test_features, testing_labels, model):
    y_pred = model(test_features)
    y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
    accuracy = accuracy_score(testing_labels, y_pred)
    precision = precision_score(testing_labels, y_pred),
    recall =recall_score(testing_labels, y_pred)
    f1 = f1_score(testing_labels, y_pred)
    print('Sum=', y_pred.sum())
    print('Accuracy', accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    return accuracy, f1
