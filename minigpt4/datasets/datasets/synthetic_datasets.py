import cv2
import torch
import numpy as np

def to_point(coord):
    return [ int(c) for c in coord ]

def draw_shape(image, shape, coord, size, color):
    """ draw shape in-place in image, centered at coord (x, y) """
    span = size / 2
    if shape == 'circle':
        cv2.circle(image, to_point(coord), int(span), color, thickness=-1)
    elif shape == 'square':
        p1 = to_point((coord[0] - span, coord[1] - span))
        p2 = to_point((coord[0] + span, coord[1] + span))
        cv2.rectangle(image, p1, p2, color, thickness=-1)
    else:
        raise NotImplementedError

def random_color():
    color_idx = np.random.choice(3)
    color = [0, 0, 0]
    color[color_idx] = 255
    return color

class CountShapesDataset(torch.utils.data.Dataset):
    def __init__(self, image_size=(224, 224), grid_size=16, shapes=('circle', 'square'), min_size=10, max_size=16, sample_rate=0.2):
        self.image_size = image_size
        self.grid_size = grid_size
        self.shapes = shapes
        self.min_size = min_size
        self.max_size = max_size
        self.sample_rate = sample_rate

        assert min_size <= max_size <= grid_size <= min(image_size)
        self.num_rows = image_size[0] // grid_size
        self.num_cols = image_size[1] // grid_size

    def __iter__(self):
        return self
    
    def __next__(self):
        """ NOTE: image channel is BGR by default in cv2 """
        image = np.zeros(shape=(self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        objects = []

        # iterate over grids, sample object per-grid
        for row in range(self.num_rows):
            center_y = self.grid_size * (row + 0.5)
            for col in range(self.num_cols):
                center_x = self.grid_size * (col + 0.5)
                if np.random.random() < self.sample_rate:
                    shape = np.random.choice(self.shapes)
                    size = np.random.uniform(self.min_size, self.max_size)
                    color = random_color()
                    draw_shape(image, shape, (center_x, center_y), size, color)
                    objects.append((shape, color))

        return image, objects
