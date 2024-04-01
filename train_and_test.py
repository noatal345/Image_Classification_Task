import numpy as np
import torch


def train(model, train_loader, criterion, optimizer, num_epochs, batch_size):
    print("Training the model...")
    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        for x_img, y_label in train_loader:
            optimizer.zero_grad()

            y_pred = model(x_img)

            loss = criterion(y_pred, y_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            prediction = y_pred.argmax(dim=1)
            correct += (prediction == y_label).sum().item()
            total += batch_size

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Loss: {round(train_loss/len(train_loader), 2)}, "
              f"Accuracy: {round(100 * correct/total, 2)}")
    return model


def train_and_test_analysis(model, train_loader, test_loader, criterion, optimizer, num_epochs, batch_size):
    print("Training the model...")
    model.train()
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for x_img, y_label in train_loader:
            optimizer.zero_grad()

            y_pred = model(x_img)

            loss = criterion(y_pred, y_label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            prediction = y_pred.argmax(dim=1)
            correct += (prediction == y_label).sum().item()
            total += batch_size

        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss}, "
              f"Train Accuracy: {round(100 * correct / total, 2)}")

        # Testing phase
        test_loss, test_acc = test_analysis(model, test_loader, criterion, batch_size)
        test_losses.append(test_loss)

    return train_losses, test_losses


def test_analysis(model, test_loader, criterion, batch_size):
    print("Testing the model...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for x_img, y_label in test_loader:
        y_pred = model(x_img)
        loss = criterion(y_pred, y_label)
        test_loss += loss.item()

        prediction = y_pred.argmax(dim=1)
        correct += (prediction == y_label).sum().item()
        total += batch_size

    test_loss /= len(test_loader)
    test_acc = 100 * correct / total

    print("Test Loss:", round(test_loss, 2))
    print("Test Accuracy:", round(test_acc, 2))

    return test_loss, test_acc


def test(model, test_loader, criterion, batch_size):
    print("Testing the model...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    for x_img, y_label in test_loader:
        y_pred = model(x_img)
        loss = criterion(y_pred, y_label)
        test_loss += loss.item()

        prediction = y_pred.argmax(dim=1)
        correct += (prediction == y_label).sum().item()
        total += batch_size

    test_acc = round(100 * correct/total, 2)
    print("Test Loss:", round(test_loss/len(test_loader), 2))
    print("Test Accuracy:", test_acc)
    return test_acc


def test_misclassification(model, test_loader, criterion, batch_size):
    print("Testing the model...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    misclassified_samples = []

    for x_img, y_label in test_loader:
        y_pred = model(x_img)
        loss = criterion(y_pred, y_label)
        test_loss += loss.item()

        prediction = y_pred.argmax(dim=1)
        correct += (prediction == y_label).sum().item()
        total += batch_size

        misclassified_indexes = (prediction != y_label).nonzero()
        for ind in misclassified_indexes:
            misclassified_samples.append((x_img[ind], y_label[ind], prediction[ind]))

    print("Test Loss:", round(test_loss/len(test_loader), 2))
    print("Test Accuracy:", round(100 * correct/total, 2))
    return misclassified_samples


def test_precision_recall(model, test_loader, criterion, batch_size):
    print("Testing the model...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    predictions = []
    labels = []

    for x_img, y_label in test_loader:
        y_pred = model(x_img)
        loss = criterion(y_pred, y_label)
        test_loss += loss.item()

        prediction = y_pred.argmax(dim=1)
        predictions.extend(prediction.tolist())
        labels.extend(y_label.tolist())

        correct += (prediction == y_label).sum().item()
        total += batch_size

    # Convert predictions and labels to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Calculate precision and recall
    precision = calculate_precision(predictions, labels)
    recall = calculate_recall(predictions, labels)

    test_acc = round(100 * correct/total, 2)
    return test_acc, precision, recall


def calculate_precision(predictions, labels):
    true_positives = ((predictions == 1) & (labels == 1)).sum()
    false_positives = ((predictions == 1) & (labels == 0)).sum()

    if true_positives + false_positives == 0:
        return 0  # Prevent division by zero
    else:
        return true_positives / (true_positives + false_positives)


def calculate_recall(predictions, labels):
    true_positives = ((predictions == 1) & (labels == 1)).sum()
    false_negatives = ((predictions == 0) & (labels == 1)).sum()

    if true_positives + false_negatives == 0:
        return 0  # Prevent division by zero
    else:
        return true_positives / (true_positives + false_negatives)