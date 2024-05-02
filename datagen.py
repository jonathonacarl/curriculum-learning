import cv2
import numpy as np
import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# labels for multi-class classification task
ELLIPSE = 0
CIRCLE = 0
RECTANGLE = 2
SQUARE = 2
TRIANGLE = 1
NON_EQ_TRIANGLE = 1


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

    triangle_height = np.random.randint(image_size // 4, image_size // 2)
    x1 = np.random.randint(0, image_size - triangle_height)
    y1 = np.random.randint(0, image_size - triangle_height)
    x2 = x1 + triangle_height
    y2 = y1
    x3 = x1 + (triangle_height // 2)
    y3 = y1 + triangle_height
    pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
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

    x1 = np.random.randint(0, image_size - side_a - 1)
    y1 = np.random.randint(0, image_size - side_b - 1)
    x2 = x1 + side_a
    y2 = y1
    x3 = np.random.randint(x1 + side_b, x1 + side_a + side_c)
    y3 = y1 + side_b

    pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)

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

    def __init__(self, dataset, labels, num_classes):
        """
        Initializes the ShapeDataset object.

        Parameters:
            dataset (torch.Tensor): A tensor containing the input data (e.g., images).
            labels (torch.Tensor): A tensor containing the corresponding labels for the input data.
            num_classes (int): The number of classes in the classification task.
        """
        self.data = dataset
        self.labels = labels
        self.num_classes = num_classes
        self.num_samples = dataset.shape[0]

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
        return self.data[index], self.labels[index]


def prepare_data(dataset, labels, batch_size, num_classes, shuffle=False):
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

    train_dataset = ShapeDataset(
        dataset=X_train, labels=y_train, num_classes=num_classes)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_dataset = ShapeDataset(
        dataset=X_val, labels=y_val, num_classes=num_classes)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)

    test_dataset = ShapeDataset(
        dataset=X_test, labels=y_test, num_classes=num_classes)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader
