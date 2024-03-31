import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from data import *
from model import *
from train_and_test import *


transform = transforms.Compose([
    transforms.ToTensor(),
])

# Define separate datasets for train and test data
train_dataset = FaceDataset(data_path='faces/face.train/train', transform=transform, shape=(19, 19))
test_dataset = FaceDataset(data_path='faces/face.test/test', transform=transform, shape=(19, 19))

# # Analyze the datasets
# train_num_images, train_image_avg_shapes, train_class_distribution = analyze_dataset(train_dataset)
# test_num_images, test_image_avg_shapes, test_class_distribution = analyze_dataset(test_dataset)
#
# print("Train dataset contains:")
# print("Total:", train_num_images, "images, of Shape:", train_image_avg_shapes, "class distribution:", train_class_distribution)
# print("Test dataset contains:")
# print("Total:", test_num_images, "images, of Shape:", test_image_avg_shapes, "class distribution:", test_class_distribution)

batch_size = 32

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# model = train(model, train_loader, criterion, optimizer, num_epochs=10, batch_size=batch_size)


def mis_classification(misclassified_samples):
    # Visualize mis-classifications
    num_subplots = min(len(misclassified_samples), 9)
    for i in range(num_subplots):
        image, true_label, predicted_label = misclassified_samples[i]
        plt.subplot(3, 3, i+1)
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label}')
        plt.axis('off')

    plt.show()


# misclassified_samples = test_misclassification(model, test_loader, criterion, batch_size=batch_size)


def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()


train_losses, test_losses = train_and_test_analysis(model, train_loader, test_loader, criterion, optimizer, 20, 32)

# Plot the training and test loss curves
plot_loss(train_losses, test_losses)
