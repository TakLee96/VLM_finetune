import cv2
import torch
import numpy as np
from PIL import Image
from collections import Counter


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
    """ NOTE: color_idx is BGR """
    colors = [ 'blue', 'green', 'red' ]
    color_idx = np.random.choice(3)
    color = [0, 0, 0]
    color[color_idx] = 255
    return color, colors[color_idx]

class CountShapesDataset(torch.utils.data.Dataset):
    def __init__(self, vis_processor=None, text_processor=None, image_size=(448, 448), grid_size=64, shapes=('circle', 'square'), min_size=24, max_size=48, sample_rate=0.2):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.image_size = image_size
        self.grid_size = grid_size
        self.shapes = shapes
        self.min_size = min_size
        self.max_size = max_size
        self.sample_rate = sample_rate

        assert min_size <= max_size <= grid_size <= min(image_size)
        self.num_rows = image_size[0] // grid_size
        self.num_cols = image_size[1] // grid_size

    # NOTE: current implementation requires dictionary style dataset
    def __len__(self):
        return 10000

    def __getitem__(self, index):
        return self.__next__(index)

    def __iter__(self):
        return self
    
    def __next__(self, seed=None):
        """ NOTE: image channel is BGR by default in cv2 """
        image = np.zeros(shape=(self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
        count = Counter()

        if seed is not None:
            np.random.seed(seed)

        # iterate over grids, sample object per-grid
        for row in range(self.num_rows):
            center_y = self.grid_size * (row + 0.5)
            for col in range(self.num_cols):
                center_x = self.grid_size * (col + 0.5)
                if np.random.random() < self.sample_rate:
                    shape = np.random.choice(self.shapes)
                    size = np.random.uniform(self.min_size, self.max_size)
                    color, color_name = random_color()
                    draw_shape(image, shape, (center_x, center_y), size, color)
                    count[(shape, color_name)] += 1

        image = image[::-1]
        if self.vis_processor is not None:
            image = self.vis_processor(Image.fromarray(image))
        
        shape = np.random.choice(self.shapes)
        _, color_name = random_color()
        question = f"how many {color_name} {shape} are there in this image?"
        
        return {
            "image": image,
            "instruction_input": f"<Img><ImageHere></Img> {question} ",
            "answer": str(count.get((shape, color_name), 0)),
        }
