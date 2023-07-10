import numpy as np


def prepare_image(image: np.ndarray, x: int, y: int, width: int, height: int,
                  size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # specify the area for pixelation
    pixelated_image = np.array(image, copy=True)     # create a copy, which will be the end product
    target_array = image[:, y: y+height, x: x+width]      # takes the area from y to y + height and x to
    # x + width and stores the values in the target_array
    average_pixel_array = np.array(target_array, copy=True)     # creates a copy of target_array to store the
    # average pixel values in it
    width_max = target_array.shape[2]   # gives endpoint on x-axis where block should stop
    height_max = target_array.shape[1]  # gives endpoint on y-axis where block should stop

    # go through all blocks of size * size for each row
    for width_ in range(0, width_max, size):
        width_end = width_ + size  # add size to the width start
        for height_ in range(0, height_max, size):
            height_end = height_ + size   # add size to the height start

            if height_end > height_max:     # if end variable would exceed the target area
                height_end = height_max     # set it to be the max amount instead
            if width_end > width_max:
                width_end = width_max

            # get the block for pixelation
            block = average_pixel_array[:, height_:height_end, width_:width_end]
            # compute the average_value of that block and store the result in pixel_array
            average_pixel = np.mean(block, axis=(1, 2))
            average_pixel_array[:, height_:height_end, width_:width_end] = average_pixel
    # overwrite the values of the pixelated_image with the average pixelated values to get the real pixel image
    pixelated_image[:, y: y + height, x: x + width] = average_pixel_array
    # create a boolean mask for the changed values in original image and the new pixelated one
    known_array = np.array(image, copy=True)   #
    known_array = np.full(known_array.shape, False)  # creates an image that is the same
    known_array[:, y: y+height, x: x+width] = True     # overwrites the Values with False in given area

    return pixelated_image, known_array, target_array

"""
    # Visualize the three images
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(pixelated_image[0], cmap='gray')
    axes[0].set_title('Pixelated Image')
    axes[1].imshow(known_array[0], cmap='gray')
    axes[1].set_title('Known Array')
    axes[2].imshow(new_target_array[0], cmap='gray')
    axes[2].set_title('New Target Array')
    plt.show()
"""

