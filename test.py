import numpy as np
import torch
from heatmap_process import calc_iterations
from Semantic_analysis import cam
from Semantic_analysis.utils import _transformer
from heatmap_process import image_to_patches
from RNN import encoder, decoder
from RNN import metric
from torchvision import transforms
import os
from RNN import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_image(image_t, output_path, num_batch):
    cam_output_path = cam.getCam(image_t, gpu=True)
    iterations, semantic_level_per_block = calc_iterations(cam_output_path)
    patches, size_orig_0, size_orig_1 = image_to_patches(image_t)
    num_rows, num_cols = encoder.encode(patches, iterations)
    ssim_per_block = decoder.decode(num_rows=num_rows, num_cols=num_cols, patches=patches, iterations=iterations,
                                    output_path=output_path, size_orig_0=size_orig_0, size_orig_1=size_orig_1)
    sissim_vector = ssim_per_block * semantic_level_per_block
    sissim = torch.sum(sissim_vector).item()
    print("si-ssim in batch {}: ".format(num_batch), sissim)
    image_t = (image_t.numpy().clip(0, 1) * 255.0).transpose(0, 2, 3, 1)
    msssim = metric.msssim(image_t, output_path)
    print("ms-ssim in batch {}: ".format(num_batch), msssim)
    return sissim, msssim

def test(directory_path, output_directory_path):
    directory_size = 24.0 # os.path.getsize(directory_path)
    print("directory_size ",directory_size)
    test_set = dataset.ImageFolder(root=directory_path, transform=_transformer())
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=4)
    sissim_total = 0
    msssim_total = 0

    for batch, data in tqdm(enumerate(test_loader)):
        filename = "kodak_{}.jpg".format(batch)
        output_path = os.path.join(output_directory_path, filename)
        sissim, msssim = test_image(data, output_path, batch)
        sissim_total += sissim
        msssim_total += msssim

    return sissim_total/directory_size, msssim_total/directory_size
