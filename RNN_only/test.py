from tqdm import tqdm
import os
from encoder import encode
from decoder import decode
from RNN import metric
import dataset
from torchvision import transforms
from torch.utils.data import DataLoader


def test_image(image_t, output_path, num_batch, item):
    encode(image_t , iters=item, output_name = str(num_batch))
    decode(orig_size=(image_t.shape[2], image_t.shape[3]),
                  output_path=output_path,
                  encoder_output_name=str(num_batch))
    image_t = (image_t.numpy().clip(0, 1) * 255.0).transpose(0, 2, 3, 1)
    msssim_result = metric.msssim_func(image_t, output_path)
    # print("ms-ssim in batch {}: ".format(num_batch), msssim_result)
    return msssim_result


def test(directory_path, output_directory_path, item):
    test_set = dataset.ImageFolder(root=directory_path,
                                   transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=4)
    msssim_list = []
    result_sizes = []

    for batch, data in tqdm(enumerate(test_loader)):
        filename = "kodak_{}_k{}.jpg".format(batch, item)
        output_path = os.path.join(output_directory_path, filename)
        print(output_path)
        msssim_results = test_image(data, output_path, batch, item)
        msssim_list.append(msssim_results)
        result_sizes.append(os.path.getsize(output_path)//1024)

    return msssim_list, result_sizes
