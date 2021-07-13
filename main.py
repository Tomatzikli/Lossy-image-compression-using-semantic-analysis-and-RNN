import numpy as np
from test import test

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
        sissim, msssim = test(directory_path='kodak', output_directory_path='kodak_test_results')
        print("sissim result: ", sissim)
        print("msssim result: ", msssim)