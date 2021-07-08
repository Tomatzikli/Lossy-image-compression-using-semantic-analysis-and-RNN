import numpy as np
import torch
from heatmap_process import calc_iterations
from Semantic_analysis import cam
from heatmap_process import image_to_patches
from RNN import encoder, decoder
from RNN.metric import msssim

def compression_allocation(cam):
        arr = cam* 255.0 / cam.max()  # gray_scale
        k = 8
        V = np.add.reduceat(np.add.reduceat(arr, np.arange(0, arr.shape[0], k), axis=0),
                            np.arange(0, arr.shape[1], k), axis=1)   # the sum of each 8x8 block.
        # print("V: ", V.shape)  # 39x32
        L = V / np.sum(arr)
        k_bar = np.mean(arr)  # 117.33
        N = V.shape[0] * V.shape[1]  # num of blocks
        a = k_bar * L * N
        b = np.ones_like(V) * k_bar
        T = np.minimum(a, b)  # not to pass k_bar. T represents num of iterations for each block

        print(T.max(), T.min(), T)  ## 117, 6
        diff = T - np.ones_like(T) * 24
        E = np.sum(np.maximum(diff, 0))

        # divide E
        # return num iterations for each block as a list


if __name__ == '__main__':
        image_path = "img1.jpg"
        cam_output_path = cam.getCam(image_path, gpu=False)
        #cam_output_path = 'cam_class__suit_prob__0.2956.jpg'
        #print("sematic ended")
        iterations, semantic_level_per_block = calc_iterations(cam_output_path)
        #print("size iterations: ", len(iterations))
        patches, size_orig_0, size_orig_1 = image_to_patches(image_path)
        #print(patches.shape)
        num_rows, num_cols = encoder.encode(patches, iterations)
        #num_rows = 31
        #num_cols = 25
        ssim_per_block = decoder.decode(num_rows, num_cols, patches, iterations, size_orig_0, size_orig_1)
        sissim =  ssim_per_block * semantic_level_per_block
        sum_sissim = torch.sum(sissim).item()
        print("si-ssim: ", sum_sissim)
        output_path = 'checkpoint/watch-join.png'
        print("ms-ssim: ", msssim(image_path, output_path))