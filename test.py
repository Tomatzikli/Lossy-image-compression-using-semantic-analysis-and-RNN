import torch
from heatmap_process import calc_iterations
# from Semantic_analysis_trained import cam
from Semantic_analysis import cam
from RNN import encoder, decoder
from RNN import metric
from torchvision import transforms
import os
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import BatchDivision


def test_image(image_t, output_path, num_batch, item):
    cam_output_path = cam.getCam(image_t, gpu=True)
    # cam_output_path = cam.getCam(image_t, network=item, gpu=True)
    iterations, semantic_level_per_block = calc_iterations(cam_output_path, mean_k=item)
    #iterations, semantic_level_per_block = calc_iterations(image_t, cam_output_path)
    batches = BatchDivision(image_t, iterations)
    encoder.encode(batches)
    ssim_per_block = decoder.decode(batches, orig_size=(
        image_t.shape[2], image_t.shape[3]), output_path=output_path)
    sissim_vector = ssim_per_block * semantic_level_per_block
    sissim = torch.sum(sissim_vector).item()
    print("si-ssim in batch {}: ".format(num_batch), sissim)
    image_t = (image_t.numpy().clip(0, 1) * 255.0).transpose(0, 2, 3, 1)
    msssim = metric.msssim(image_t, output_path)
    psnr = metric.psnr(image_t, output_path)
    print("ms-ssim in batch {}: ".format(num_batch), msssim)
    return sissim, msssim, psnr


def test(directory_path, output_directory_path, item):
    test_set = dataset.ImageFolder(root=directory_path,
                                   transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=4)
    sissim_list = []
    msssim_list = []
    psnr_list = []

    for batch, data in tqdm(enumerate(test_loader)):
        if batch <= 9:
            filename = "kodak_0{}_k{}.jpg".format(batch, item)
        else:
            filename = "kodak_{}_k{}.jpg".format(batch, item)
        output_path = os.path.join(output_directory_path, filename)
        sissim, msssim, psnr = test_image(data, output_path, batch, item)
        sissim_list.append(sissim)
        msssim_list.append(msssim)
        psnr_list.append(psnr)

    return sissim_list, msssim_list, psnr_list


def results_from_storage(directory_path, output_directory_path, item):
    print("cude available? ", torch.cuda.is_available())
    # changing the name of the photos a format we can traverse
    """
    for root, _, fnames in sorted(os.walk(output_directory_path, followlinks=True)):
        i = 0
        for fname in sorted(fnames):
            first_num = fname.split('_')[1]
            second_num = fname.split("_k")[1].split(".")[0]
            if len(first_num) == 1:
                first_num = "0"+first_num
            if len(second_num) == 1:
                second_num = "0"+second_num
            print(first_num + "," + second_num)
            new_name = "kodak_{}_k{}.jpg".format(first_num, second_num)
            os.rename(os.path.join(output_directory_path, fname),
                      os.path.join(output_directory_path, new_name))
    for root, _, fnames in sorted(os.walk(output_directory_path, followlinks=True)):
        i = 0
        for fname in sorted(fnames):
            print(fname)
    """
    test_set = dataset.ImageFolder(root=directory_path,
                                   transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=4)
    result_set = dataset.ImageFolder(root=output_directory_path,
                                   transform=transforms.ToTensor())
    indices = list(range(item-1, len(result_set), 30)) # 30 is the total number of K!
    result_set = torch.utils.data.Subset(result_set, indices) # take only pics of a certain K
    sissim_list = []
    msssim_list = []
    result_sizes = []

    for i in range(len(test_set)):
        test_num = "{}".format(i+1)
        pic_num = "{}".format(i)
        k_num = "{}".format(item)
        if len(test_num) == 1:
            test_num = "0" + test_num
        if len(pic_num) == 1:
            pic_num = "0" + pic_num
        if len(k_num) == 1:
            k_num = "0" + k_num
        filename = "kodak_{}_k{}.jpg".format(pic_num, k_num)
        # print("{}".format(filename))
        output_path = os.path.join(output_directory_path, filename)
        iterations, semantic_level_per_block = calc_iterations(test_set[i],
                                                               cam.getCam(test_set[i].unsqueeze(0), gpu=True),
                                                               mean_k=item)
        ssim_per_block = []
        test_patches = test_set[i].squeeze().data.unfold(0, 3, 3).unfold(
            1, 8, 8).unfold(2, 8, 8).squeeze()
        result_patches = result_set[i].squeeze().data.unfold(0, 3, 3).unfold(
            1, 8, 8).unfold(2, 8, 8).squeeze()
        result_patches = result_patches.reshape(result_patches.shape[0]*result_patches.shape[1], 3, 8, 8)
        test_patches = test_patches.reshape(
            test_patches.shape[0] * test_patches.shape[1], 3, 8, 8)
        for j in range(len(result_patches)):
            ssim_per_block.append(ssim(test_patches[j].unsqueeze(0), result_patches[j].unsqueeze(0)))
        ssim_per_block = torch.tensor(ssim_per_block)
        sissim_vector = ssim_per_block * semantic_level_per_block
        sissim = torch.sum(sissim_vector).item()
        msssim = metric.msssim("kodak/kodim{}.png".format(test_num), output_path)
        print("{}".format(msssim))
        print("calc finished")
        sissim_list.append(sissim)
        msssim_list.append(msssim)
        result_sizes.append(os.path.getsize(output_path) // 1024)

    return sissim_list, msssim_list, result_sizes
