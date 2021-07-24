import numpy as np
import torch
from heatmap_process import calc_iterations
from Semantic_analysis_trained import cam
from RNN import encoder, decoder
from RNN import metric
from torchvision import transforms
import os
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def test_image(image_t, output_path, num_batch, item):
    cam_output_path = cam.getCam(image_t, gpu=True)
    iterations, semantic_level_per_block = calc_iterations(cam_output_path, mean_k=item)
    patches_data_loader = encoder.encode(image_t, iterations)
    ssim_per_block = decoder.decode(patches_data_loader.dataset, orig_size=(image_t.shape[2], image_t.shape[3]),
                                    iterations=iterations, output_path=output_path)
    sissim_vector = ssim_per_block * semantic_level_per_block
    sissim = torch.sum(sissim_vector).item()
    print("si-ssim in batch {}: ".format(num_batch), sissim)
    image_t = (image_t.numpy().clip(0, 1) * 255.0).transpose(0, 2, 3, 1)
    msssim = metric.msssim(image_t, output_path)
    print("ms-ssim in batch {}: ".format(num_batch), msssim)
    return sissim, msssim

def test(directory_path, output_directory_path, item):
    print("cude available? ", torch.cuda.is_available())
    test_set = dataset.ImageFolder(root=directory_path, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=4)
    sissim_list = []
    msssim_list = []
    cnt = 0

    for batch, data in tqdm(enumerate(test_loader)):
        filename = "kodak_{}_k{}.jpg".format(batch, item)
        output_path = os.path.join(output_directory_path, filename)
        print(output_path)
        sissim, msssim = test_image(data, output_path, batch, item)
        sissim_list.append(sissim)
        msssim_list.append(msssim)
        cnt += 1
        if cnt == 8:
            break

    return sissim_list, msssim_list
