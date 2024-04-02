from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from data import *
from model import *
from train_and_test import *


# This function trains the model and returns the test accuracy
# The function is called by the wandb agent
def img_classifier(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Define separate datasets for train and test data
    train_dataset = FaceDataset(data_path='faces/face.train/train', transform=transform, shape=(19, 19))
    test_dataset = FaceDataset(data_path='faces/face.test/test', transform=transform, shape=(19, 19))

    batch_size = config.batch_size

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    model = CNN()

    # Define the loss function, optimizer and learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Train and test the model
    model = train(model, train_loader, criterion, optimizer, num_epochs=config.epoch, batch_size=batch_size)
    test_accuracy = test(model, test_loader, criterion, batch_size=batch_size)
    return test_accuracy

