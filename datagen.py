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

    dataset, labels = [], []
    for _ in range(num_examples):
        shape_image, shape_type = generate_shape(image_size, complex=complex)
        shape_type = shape_label(shape_type)
        dataset.append(shape_image)
        labels.append(shape_type)

    return tensor(dataset, dtype=torch.float32), tensor(labels, dtype=torch.long)


class ShapeDataset(Dataset):
    def __init__(self, dataset, labels, num_classes):
        self.data = dataset
        self.labels = labels
        self.num_classes = num_classes
        self.num_samples = dataset.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def prepare_data(dataset, labels, batch_size, num_classes, shuffle=False):

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
