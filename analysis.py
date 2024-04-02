import random
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from data import *
from model import *
from train_and_test import *


# This file contains functions to analyze the datasets such as:
# visualize mis-classifications
# calculate the average number of misclassified samples over 10 runs.
# calculate the average precision and recall of the model over 10 runs.
# visualize the feature maps of the conv layers.
def data_analysis():
    # Analyze the datasets before training the model
    train_num_images, train_image_avg_shapes, train_class_distribution = analyze_dataset(train_dataset)
    test_num_images, test_image_avg_shapes, test_class_distribution = analyze_dataset(test_dataset)

    print("Train dataset contains:")
    print("Total:", train_num_images, "images, of Shape:", train_image_avg_shapes, "class distribution:", train_class_distribution)
    print("Test dataset contains:")
    print("Total:", test_num_images, "images, of Shape:", test_image_avg_shapes, "class distribution:", test_class_distribution)


def mis_classification(misclassified_samples):
    # Visualize of 9 mis-classifications images from the test set
    num_subplots = min(len(misclassified_samples), 9)
    for i in range(num_subplots):
        image, true_label, predicted_label = misclassified_samples[i]
        plt.subplot(3, 3, i+1)
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')

    plt.show()


def avg_misclassified(model, train_loader, test_loader, criterion, optimizer, batch_size=32):
    print("Calculating the average number of mis-classified samples...")
    # calculate the average number of mis-classified samples for both classes over 10 runs
    total_face = []
    total_non_face = []

    for i in range(10):
        # train the model each time
        model = train(model, train_loader, criterion, optimizer, num_epochs=10, batch_size=batch_size)

        misclassified_samples = test_misclassification(model, test_loader, criterion, batch_size)
        random.shuffle(misclassified_samples)
        # sample[1] is the true label, sample[2] is the predicted label
        non_face = 0
        face = 0
        for sample in misclassified_samples:
            if sample[1] == 0 and sample[2] == 1:
                non_face += 1
            elif sample[1] == 1 and sample[2] == 0:
                face += 1
        # print(f"Non-face mis-classified as face: {non_face}")
        # print(f"Face mis-classified as non-face: {face}")
        total_face.append(face)
        total_non_face.append(non_face)
        # mis_classification(misclassified_samples)

    print("Average number of mis-classified samples non-face:", sum(total_non_face)/len(total_non_face))
    print("Average number of mis-classified samples face:", sum(total_face)/len(total_face))


def avg_pr(model, train_loader, test_loader, criterion, optimizer, batch_size=32):
    print("Calculating the average precision and recall of the model...")
    # calc the average precision and recall of the model over 10 runs(32 batch size, 10 epochs, 0.001 learning rate)
    total_precision = []
    total_recall = []
    for i in range(10):
        model = train(model, train_loader, criterion, optimizer, num_epochs=10, batch_size=batch_size)
        test_acc, precision, recall = test_precision_recall(model, test_loader, criterion, batch_size)
        total_precision.append(precision)
        total_recall.append(recall)
        print("Test accuracy:", test_acc)
        # print("Precision:", precision)
        # print("Recall:", recall)

    print("Average Precision:", sum(total_precision) / len(total_precision))
    print("Average Recall:", sum(total_recall) / len(total_recall))


def plot_loss(train_losses, test_losses):
    # Plot the training and test loss curves to check for overfitting
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()


def visualize_feature_maps(model, input_image_tensor):

    model.eval()
    print("Visualizing feature maps...")
    # Forward pass the input image through the model
    with torch.no_grad():
        # Add a batch dimension to the input image
        features = input_image_tensor.unsqueeze(0)
        # Iterate through the convolutional layers of the model
        for layer in model.children():
            if isinstance(layer, torch.nn.Conv2d):
                features = layer(features)
                # Extract feature maps from convolutional layers
                feature_map = features.squeeze(0).detach().cpu().numpy()
                # Get the number of filters in the current layer
                num_filters = feature_map.shape[0]

                # Plot the feature maps for the current layer
                fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
                fig.tight_layout(pad=3.0)

                # Iterate through each filter and plot its feature map
                for i in range(num_filters):
                    ax = axes[i]
                    ax.imshow(feature_map[i], cmap='viridis')
                    ax.axis('off')

                # Show the plot for the current layer
                plt.show()


# init program without wandb
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

# Define separate datasets for train and test data
train_dataset = FaceDataset(data_path='faces/face.train/train', transform=transform, shape=(19, 19))
test_dataset = FaceDataset(data_path='faces/face.test/test', transform=transform, shape=(19, 19))


batch_size = 32

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
model = train(model, train_loader, criterion, optimizer, num_epochs=10, batch_size=batch_size)


# In This Part of the code you can choose what you want to analyze and put the rest in comments

# # Check for overfitting by plotting the training and test loss curves
# train_losses, test_losses = train_and_test_analysis(model, train_loader, test_loader, criterion, optimizer, 30, 32)
# plot_loss(train_losses, test_losses)
#
# # Check for the avg mis-classifications
# avg_misclassified(model, train_loader, test_loader, criterion, optimizer, batch_size)

# Visualize the feature maps of the convolutional layers on the first image in the test dataset
# input_image, _ = test_dataset[0]
# visualize_feature_maps(model, input_image)
#
#
# # Check for the avg precision and recall
# avg_pr(model, train_loader, test_loader, criterion, optimizer, batch_size)

