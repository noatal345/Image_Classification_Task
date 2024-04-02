from torch import nn, sigmoid


# This is the CNN model that will be used to classify the images
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(0.25)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 output classes: face or non-face

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.bn1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.bn2(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.bn3(x)
        x = self.pool(x)

        x = x.view(-1, 128 * 2 * 2)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        x = sigmoid(x)
        return x
