from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


# This is the dataset class that will be used to load the images
class FaceDataset(Dataset):
    def __init__(self, data_path, transform=None, shape=(19, 19)):
        self.data = data_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_data(shape)

    def load_data(self, shape=(19, 19)):
        # This function loads the images and labels from the faces directory
        # The images can be resized to the specified shape

        for folder_name in os.listdir(self.data):
            if folder_name == 'face':
                label = 1
            else:
                label = 0
            for file in os.listdir(os.path.join(self.data, folder_name)):
                image = Image.open(os.path.join(self.data, folder_name, file))
                image = image.resize((shape[0], shape[1]))
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # convert image to tensor
        if self.transform:
            image = self.transform(image)

        return image, label


def calc_images_avg_size(path):
    # calculate the average size of the images in the dataset
    total_width = 0
    total_height = 0
    total_images = 0
    for folder_name in os.listdir('faces/face.'+path):
        for file in os.listdir(os.path.join('faces/face.'+path, folder_name)):
            image = Image.open(os.path.join('faces/face.'+path, folder_name, file))
            width, height = image.size
            total_width += width
            total_height += height
            total_images += 1

    avg_width = total_width / total_images
    avg_height = total_height / total_images
    print("Average Width:", avg_width)
    print("Average Height:", avg_height)


def analyze_dataset(dataset):
    # this function should return:
    # the number of images, the average shapes of the images, and the class distribution of the dataset
    num_images = len(dataset)
    image_shapes = [image[0].shape for image in dataset]
    images_avg_shape = np.mean(image_shapes, axis=0)
    # calculate the class distribution of the dataset
    class_distribution = {0: 0, 1: 0}
    for _, label in dataset:
        class_distribution[label] += 1
    class_distribution = {"non-face": class_distribution[0], "face": class_distribution[1]}
    return num_images, images_avg_shape, class_distribution

