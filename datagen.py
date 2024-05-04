import cv2
import random
import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split

# labels for multi-class classification task
ELLIPSE = 0
CIRCLE = 0
TRIANGLE = 1
NON_EQ_TRIANGLE = 1
RECTANGLE = 2
SQUARE = 2


def shape_label(shape_type):
    """
    Description:
        This function converts a shape type string into an integer label for multi-class classification.

    Parameters:
        shape_type (str): A string representing the shape type (e.g., 'circle', 'square', 'triangle', etc.).

    Returns:
        int: An integer label corresponding to the shape type.

    Notes:
        The function uses global constants (ELLIPSE, CIRCLE, RECTANGLE, SQUARE, TRIANGLE, NON_EQ_TRIANGLE)
        to map the string shape type to an integer label.
    """
    if shape_type == 'circle':
        shape_type = CIRCLE
    elif shape_type == 'ellipse':
        shape_type = ELLIPSE
    elif shape_type == 'square':
        shape_type = SQUARE
    elif shape_type == 'rectangle':
        shape_type = RECTANGLE
    elif shape_type == 'eq_triangle':
        shape_type = TRIANGLE
    else:
        shape_type = NON_EQ_TRIANGLE

    return shape_type


def generate_eq_triangle(image_size):
    """
    Description:
        This function generates the coordinates of an equilateral triangle within the given image size.

    Parameters:
        image_size (int): The size of the square image.

    Returns:
        numpy.ndarray: A 3x2 NumPy array containing the (x, y) coordinates of the three vertices of the equilateral triangle.

    Notes:
        The function generates a random triangle height within a certain range and calculates the coordinates
        of the three vertices based on the triangle height and random starting positions.
    """
    while True:
        triangle_height = np.random.randint(image_size // 4, image_size // 2)
        x1 = np.random.randint(0, image_size - triangle_height)
        y1 = np.random.randint(0, image_size - triangle_height)
        x2 = x1 + triangle_height
        y2 = y1
        x3 = x1 + (triangle_height // 2)
        y3 = y1 + triangle_height
        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
        if (pts >= 0).all() and (pts[:, 0] < image_size).all() and (pts[:, 1] < image_size).all():
            return pts


def generate_non_eq_triangle(image_size):
    """
    Description:
        This function generates the coordinates of a non-equilateral triangle within the given image size.

    Parameters:
        image_size (int): The size of the square image.

    Returns:
        numpy.ndarray: A 3x2 NumPy array containing the (x, y) coordinates of the three vertices of the non-equilateral triangle.

    Notes:
        The function generates three random side lengths for the triangle, ensures that the side lengths satisfy
        the triangle inequality, and calculates the coordinates of the three vertices based on the side lengths
        and random starting positions.
    """
    while True:
        side_a = np.random.randint(image_size // 4, image_size // 2)
        side_b = np.random.randint(image_size // 4, image_size // 2)
        side_c = np.random.randint(image_size // 4, image_size // 2)

        while not (side_a + side_b > side_c and side_b + side_c > side_a and side_a + side_c > side_b):
            side_a = np.random.randint(image_size // 4, image_size // 2)
            side_b = np.random.randint(image_size // 4, image_size // 2)
            side_c = np.random.randint(image_size // 4, image_size // 2)

        while side_a == side_b or side_b == side_c or side_a == side_c:
            side_a = np.random.randint(image_size // 4, image_size // 2)
            side_b = np.random.randint(image_size // 4, image_size // 2)
            side_c = np.random.randint(image_size // 4, image_size // 2)

        x1 = np.random.randint(0, image_size - max(side_a, side_c) - 1)
        y1 = np.random.randint(0, image_size - side_b - 1)
        x2 = x1 + side_a
        y2 = y1
        x3 = x1 + side_c
        y3 = y1 + side_b

        pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)

        if (pts >= 0).all() and (pts[:, 0] < image_size).all() and (pts[:, 1] < image_size).all():
            return pts


def generate_shape(image_size, complex=False):
    """
    Description:
        This function generates a random shape (square, circle, rectangle, ellipse, equilateral triangle, or non-equilateral triangle)
        within the given image size.

    Parameters:
        image_size (int): The size of the square image.
        complex (bool, optional): If True, more complex shapes (rectangle, ellipse, non-equilateral triangle) are included.
                                  If False (default), only simple shapes (square, circle, equilateral triangle) are included.

    Returns:
        Tuple[numpy.ndarray, str]: A grayscale image containing the generated shape and a
                                   string representing the type of the generated shape.

    Notes:
        The function randomly selects a shape type, generates the shape within the image using appropriate parameters,
        and returns the image and the shape type.
    """

    # add more complicated shapes for complex dataset
    shape_type = np.random.choice(
        ['rectangle', 'ellipse', 'non_eq_triangle']) if complex else np.random.choice(['square', 'circle', 'eq_triangle'])
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    if shape_type == 'square':
        side_length = np.random.randint(image_size // 4, image_size // 2)
        x = np.random.randint(0, image_size - side_length)
        y = np.random.randint(0, image_size - side_length)
        image[y:y+side_length, x:x+side_length] = 255

    elif shape_type == 'circle':
        radius = np.random.randint(image_size // 8, image_size // 4)
        center_x = np.random.randint(radius, image_size - radius)
        center_y = np.random.randint(radius, image_size - radius)
        cv2.circle(image, (center_x, center_y), radius, 255, -1)

    elif shape_type == 'rectangle':
        width = np.random.randint(image_size // 4, image_size // 2)
        height = np.random.randint(image_size // 4, image_size // 2)
        x = np.random.randint(0, image_size - width)
        y = np.random.randint(0, image_size - height)
        image[y:y+height, x:x+width] = 255

    elif shape_type == 'ellipse':
        major_axis = np.random.randint(image_size // 4, image_size // 2)
        minor_axis = np.random.randint(image_size // 4, image_size // 2)
        center_x = np.random.randint(major_axis, image_size - major_axis)
        center_y = np.random.randint(minor_axis, image_size - minor_axis)
        cv2.ellipse(image, (center_x, center_y),
                    (major_axis, minor_axis), 0, 0, 360, 255, -1)

    elif shape_type == 'eq_triangle':
        pts = generate_eq_triangle(image_size)
        cv2.fillPoly(image, [pts], 255)

    elif shape_type == 'non_eq_triangle':
        pts = generate_non_eq_triangle(image_size)
        cv2.fillPoly(image, [pts], 255)

    return image.astype(np.float32), shape_type


def generate_shape_dataset(num_examples, image_size, complex=False):
    """
    Description:
        This function generates a dataset of random shapes and their corresponding labels.

    Parameters:
        num_examples (int): The number of examples (images) to generate.
        image_size (int): The size of the square images.
        complex (bool, optional): If True, more complex shapes (rectangle, ellipse, non-equilateral triangle) are included.
                                  If False (default), only simple shapes (square, circle, equilateral triangle) are included.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
        - A tensor of shape (num_examples, image_size, image_size) containing the generated images
        - A tensor of shape (num_examples,) containing the corresponding labels (integers) for the images.

    Notes:
        The function calls the `generate_shape` function to generate individual images and labels,
        and collects them into PyTorch tensors.
    """
    dataset, labels = [], []
    for _ in range(num_examples):
        shape_image, shape_type = generate_shape(image_size, complex=complex)
        shape_type = shape_label(shape_type)
        dataset.append(shape_image)
        labels.append(shape_type)

    return tensor(dataset, dtype=torch.float32), tensor(labels, dtype=torch.long)


class ShapeDataset(Dataset):
    """
    A custom PyTorch Dataset class for shape classification.

    Attributes:
        data (torch.Tensor): A tensor containing the input data (e.g., images).
        labels (torch.Tensor): A tensor containing the corresponding labels for the input data.
        num_classes (int): The number of classes in the classification task.
        num_samples (int): The number of samples (images) in the dataset.

    Methods:
        __len__():
            Returns the number of samples in the dataset.

        __getitem__(index):
            Returns a sample and its corresponding label from the dataset.

    """

    def __init__(self, dataset, labels, num_classes, num_basic=0):
        """
        Initializes the ShapeDataset object.

        Parameters:
            dataset (torch.Tensor): A tensor containing the input data (e.g., images).
            labels (torch.Tensor): A tensor containing the corresponding labels for the input data.
            num_classes (int): The number of classes in the classification task.
        """
        self.dataset = dataset
        self.labels = labels
        self.num_classes = num_classes
        self.num_samples = dataset.shape[0]

        self.basic_counts = num_basic
        self.complex_counts = self.num_samples - self.basic_counts

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        Returns a sample and its corresponding label from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The input data (e.g., image) at the specified index.
                - torch.Tensor: The label corresponding to the input data at the specified index.
        """
        return self.dataset[index], self.labels[index]

    def set_num_basic(self, num_basic):
        """
        Sets number of basic or complex in the dataset.

        Args:
            num_basic (int): The number of basic shapes in the dataset.
        """

        self.basic_counts = num_basic
        self.complex_counts = self.num_samples - self.basic_counts


def prepare_data(dataset, labels, batch_size, num_classes, shuffle=False, data_type=0):
    """
    Description:
        This function splits the input dataset and labels into train, validation, and test sets,
        and creates PyTorch data loaders for each set.

    Parameters:
        dataset (torch.Tensor): A tensor containing the input data (e.g., images).
        labels (torch.Tensor): A tensor containing the corresponding labels for the input data.
        batch_size (int): The batch size for the data loaders.
        num_classes (int): The number of classes in the classification task.
        shuffle (bool, optional): If True, the data is shuffled before splitting. Default is False.
        data_type (int, optional): 0 if data is basic, 1 if data is complex, 2 if data is combined

    Returns:
        Tuple: A tuple containing:
            - torch.utils.data.DataLoader: The data loader for the training set.
            - torch.utils.data.DataLoader: The data loader for the validation set.
            - torch.utils.data.DataLoader: The data loader for the test set.

    Notes:
        The function splits the input data into train, validation, and test sets using `train_test_split` from scikit-learn.
        It then creates PyTorch datasets and data loaders for each set.
    """

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels,
                                                        train_size=0.5, test_size=0.5)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                    train_size=0.5, test_size=0.5)

    total_basic_train = len(X_train)
    total_basic_val = len(X_val)
    total_basic_test = len(X_test)

    if data_type == 1:
        total_basic_train = 0
        total_basic_val = 0
        total_basic_test = 0

    if data_type == 2:
        total_basic_train = total_basic_train // 2
        total_basic_val = total_basic_val // 2
        total_basic_test = total_basic_test // 2

    train_dataset = ShapeDataset(
        dataset=X_train, labels=y_train, num_classes=num_classes, num_basic=total_basic_train)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_dataset = ShapeDataset(
        dataset=X_val, labels=y_val, num_classes=num_classes, num_basic=total_basic_val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)

    test_dataset = ShapeDataset(
        dataset=X_test, labels=y_test, num_classes=num_classes, num_basic=total_basic_test)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader


def generate_proportional_data(loader_basic: DataLoader, loader_combined: DataLoader, curr_p: float, num_classes=3, batch_size=100, shuffle=True):
    """
    Description:
        Generates a DataLoader that combines easy and complex data with a specified probability.

    Parameters:
        loader_basic (DataLoader): DataLoader for easy data.
        loader_combined (DataLoader): DataLoader for combined (easy and complex) data.
        curr_p (float): Current probability of using combined data.
        target_p (float, optional): Target probability of using combined data (default=1.0).
        batch_size (int, optional): Batch size for the generated DataLoader (default=32).
        shuffle (bool, optional): Whether to shuffle the data during training (default=True).

    Returns:
        DataLoader: DataLoader that generates combined data with probability `curr_p` and easy data with probability `1 - curr_p`.
    """

    def get_random_subset(dataset, desired_length):

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        sample_indices = indices[:desired_length]
        random_subset = Subset(dataset, sample_indices)
        return random_subset

    # ensure models are only trained on fixed amount of data (i.e we don't add data)
    if curr_p >= 1:
        return loader_combined

    # each index of loader is a batch
    total_length = len(loader_combined) * batch_size

    # calculate the desired lengths based on the current probability
    desired_combined_length = int(curr_p * total_length)
    desired_basic_length = total_length - desired_combined_length

    basic_subset = get_random_subset(
        loader_basic.dataset, desired_basic_length)
    combined_subset = get_random_subset(
        loader_combined.dataset, desired_combined_length)

    concatted = ConcatDataset(
        [basic_subset, combined_subset])

    # combine the datasets
    proportional_dataset = DataLoader(concatted)

    data, labels = [], []
    for d, l in proportional_dataset:
        data.append(d)
        labels.append(l.item())

    dataset = torch.stack(data)
    if dataset.dim() != 3:
        dataset = dataset.flatten(start_dim=0, end_dim=-3)

    dataset_labels = tensor(labels, dtype=torch.long)
    shape_dataset = ShapeDataset(
        dataset=dataset, labels=dataset_labels, num_classes=num_classes)

    # combined shapes are a 50-50 split between basic and complex shapes
    shape_dataset.set_num_basic(
        desired_basic_length + desired_combined_length//2)

    return DataLoader(shape_dataset, batch_size=batch_size, shuffle=shuffle)
