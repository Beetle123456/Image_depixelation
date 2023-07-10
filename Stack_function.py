import torch


def stack_with_padding(batch_as_list: list):

    pixelated_image_tensors = []
    known_array_tensors = []
    original_image_tensors = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # batch_as_list[i][0] -> pixelated_image, batch_as_list[i][1] -> known_array, batch_as_list[i][2] -> original image
    for i in range(len(batch_as_list)):
        pixelated_image_tensors.append(torch.from_numpy(batch_as_list[i][0]).to(device))
        known_array_tensors.append(torch.from_numpy(batch_as_list[i][1]).to(device))
        original_image_tensors.append(torch.from_numpy(batch_as_list[i][2]).to(device))

    stacked_pixelated_images = torch.stack(pixelated_image_tensors, dim=0)
    stacked_known_arrays = torch.stack(known_array_tensors, dim=0)
    stacked_original_images = torch.stack(original_image_tensors, dim=0)

    return stacked_pixelated_images, stacked_known_arrays, stacked_original_images
