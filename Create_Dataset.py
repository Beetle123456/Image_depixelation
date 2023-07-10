import typing
import numpy as np
import os
from PIL import Image
from Prepare_Image import prepare_image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


im_size = 64
resize_transforms = transforms.Compose([
    transforms.Resize(size=im_size),
    transforms.CenterCrop(size=(im_size, im_size)),
    transforms.Grayscale(num_output_channels=1),
])


class RandomImagePixelationDataset(Dataset):

    def __init__(
            self,
            image_dir,
            dtype: typing.Optional[type] = None
    ):
        self.image_dir = image_dir
        self.dtype = dtype
        self.image_list = self.get_image_list()     # list to store the image paths

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
        with Image.open(self.image_list[index]) as image:    # opens image with index from list of images
            image = resize_transforms(image)
            image_array = np.array(image)
            original_image = np.expand_dims(image_array, axis=0)     # converts image to array and stores the array
            if self.dtype is not None:      # if dtype is specified
                original_image = original_image.astype(self.dtype)  # make the array as that type

            rng = np.random.default_rng(seed=index)  # creates a rng with seed set to index

            width = rng.integers(4, 32)  # random number generated for width in the between 4-32
            height = rng.integers(4, 32)
            size = rng.integers(4, 16)     # same as width
            x = rng.integers(0, 64 - width)   # random number between 0 and the
            y = rng.integers(0, 64 - height)   # random number between 0 and the

            pixelated_image, known_array, target_array = prepare_image(original_image, x, y, width, height, size)

        return pixelated_image, known_array, original_image

    def __len__(self):
        return len(self.image_list)
