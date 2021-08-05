import numpy as np
from test import test
import pandas as pd
import os
import torch


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

def save_results(sissim_results, msssim_results, result_sizes, orig_sizes, experiments, output_name):
        df = pd.DataFrame()
        df["orig_sizes"] = orig_sizes + [""]
        for i in range(len(experiments)):
                name_col = "sissim_{}".format(experiments[i])
                sissim_results[i] += [sum(sissim_results[i])/len(sissim_results[i])]
                df[name_col] = sissim_results[i]
                name_col = "msssim_{}".format(experiments[i])
                msssim_results[i] += [sum(msssim_results[i])/len(msssim_results[i])]
                df[name_col] = msssim_results[i]
                name_col = "result_sizes_{}".format(experiments[i])
                df[name_col] = result_sizes[i] + [""]
                sizes_ratio = torch.tensor(orig_sizes) / torch.tensor(result_sizes[i])
                sizes_ratio = torch.cat((sizes_ratio, torch.tensor([sum(sizes_ratio) / len(sizes_ratio)])))
                name_col = "sizes_ratio_{}".format(experiments[i])
                df[name_col] = sizes_ratio
        df.index = list(range(len(sissim_results[0])-1)) + ["avg"] # num of images processed
        print(df)
        df.to_excel(output_name +'.xlsx')


if __name__ == '__main__':
        sissim_all = []
        msssim_all = []
        result_sizes_all = []
        # experiments = [2, 4, 6, 8, 10, 12]
        experiments = [12]  # mean k
        for item in experiments:
                sissim, msssim,  result_sizes = test(directory_path='kodak', output_directory_path='kodak_test_results', item = item)
                print("sissim result: ", sissim)
                print("msssim result: ", msssim)
                sissim_all.append(sissim)
                msssim_all.append(msssim)
                result_sizes_all.append(result_sizes)
        print("len(sissim_all[0]", sissim_all[0])

        orig_sizes = []
        for i in range(1, 25):
                if i <= 9:
                        input_path = "kodak/kodim0{}.png".format(i)
                else:
                        input_path = "kodak/kodim{}.png".format(i)
                orig_sizes.append(os.path.getsize(input_path) // 1024)

        output_name = 'fast_patch32_k12'
        save_results(sissim_all, msssim_all, result_sizes_all, orig_sizes, experiments, output_name)