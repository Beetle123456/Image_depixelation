import typing
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from Prepare_Image import prepare_image


class RandomImagePixelationDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int],
            dtype: typing.Optional[type] = None
    ):
        self.image_dir = image_dir
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype
        self.image_list = self.get_image_list()     # list to store the image paths

        if self.width_range[0] < 2 or self.height_range[0] < 2 or self.size_range[0] < 2:
            raise ValueError("Smallest possible value for width_range, height_range and size_range can't be smaller "
                             "than 2")
        if self.width_range[0] > self.width_range[1]:
            raise ValueError("Smallest possible value for width_range can't be bigger than biggest possible value. "
                             "First insert smaller number, then bigger number")
        if self.height_range[0] > self.height_range[1]:
            raise ValueError("Smallest possible value for height_range can't be bigger than biggest possible value."
                             "First insert smaller number, then bigger number")
        if self.size_range[0] > self.size_range[1]:
            raise ValueError("Smallest possible value for size_range can't be bigger than biggest possible value."
                             "First insert smaller number, then bigger number")

    def get_image_list(self):
        if not os.path.exists(self.image_dir):  # checks if path is a valid directory
            raise ValueError(f"'{self.image_dir}' must be a valid path, check if path exists or if there is a "
                             f"spelling mistake in path")

        valid_suffix = {".jpg"}     # for checking the .jpg suffix
        image_list = []

        for dirpath, dirname, filenames in os.walk(self.image_dir):     # os.walk yields 3-tuples in directory and all
            # subdirectories
            for item in filenames:    # for each item in filenames (list of all filenames in directory and subdirectory)
                if os.path.splitext(item)[1] in valid_suffix:   # splitext splits root and suffix for checking
                    image_list.append(os.path.join(dirpath, item))  # takes the dirpath and joins it together with the
                    # item name
        image_list.sort()   # sorts list
        return image_list   # returns the list of images with absolute pathnames

    def __getitem__(self, index):
        with Image.open(self.image_list[index]) as target_image:    # opens image with index from list of images
            target_image_array = np.array(target_image)     # converts image to array and stores the array
            if self.dtype is not None:      # if dtype is specified
                target_image_array = target_image_array.astype(self.dtype)  # make the array as that type

            rng = np.random.default_rng(seed=index)  # creates a rng with seed set to index

            width = rng.integers(self.width_range[0], self.width_range[1])  # random number generated for width in the
            # boundaries of width_range
            width = min(width, target_image_array.shape[2])     # takes the minimum out of these 2 values
            height = rng.integers(self.height_range[0], self.height_range[1])   # same as width
            height = min(height, target_image_array.shape[1])   # takes the minimum out of these 2 values
            size = rng.integers(self.size_range[0], self.size_range[1])     # same as width

            if int(width) == int(target_image_array.shape[2]):
                x = 0
            else:
                x = rng.integers(0, target_image_array.shape[2] - width)   # random number between 0 and the
            # number of columns (length of one row)
            if height == target_image_array.shape[1]:
                y = 0
            else:
                y = rng.integers(0, target_image_array.shape[1] - height)   # random number between 0 and the
            # number of rows (length of column)

            converted_image = prepare_image(target_image_array, x, y, width, height, size)   # calls the function
            # with the previously created grayscale image

        return converted_image[0], converted_image[1], converted_image[2], self.image_list[index]

    def __len__(self):
        return len(self.image_list)
