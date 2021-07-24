import numpy as np
from test import test
import pandas as pd
import openpyxl

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

def save_results(sissim_results, msssim_results, experiments, output_name):
        df = pd.DataFrame()
        index = []
        for i in range(len(experiments)):
                name_col = "sissim_{}".format(experiments[i])
                sissim_results[i] += [sum(sissim_results[i])/len(sissim_results[i])]
                df[name_col] = sissim_results[i]
                name_col = "msssim_{}".format(experiments[i])
                msssim_results[i] += [sum(msssim_results[i])/len(msssim_results[i])]
                df[name_col] = msssim_results[i]

        df.index = list(range(len(sissim_results[0])-1)) + ["avg"] # num of images processed
        print(df)
        df.to_excel(output_name +'.xlsx')


if __name__ == '__main__':
        sissim_all = []
        msssim_all = []
        experiments = [4]  # mean k
        for item in experiments:
                sissim, msssim = test(directory_path='kodak', output_directory_path='kodak_test_results', item = item)
                print("sissim result: ", sissim)
                print("msssim result: ", msssim)
                sissim_all.append(sissim)
                msssim_all.append(msssim)


        output_name = 'trained_semantic_k4'
        save_results(sissim_all, msssim_all, experiments, output_name)