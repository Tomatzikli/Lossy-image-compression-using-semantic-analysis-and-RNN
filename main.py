import numpy as np
from test import test
import pandas as pd
import os
import torch


def save_results(sissim_results, msssim_results, psnr_results, experiments, output_name):
        df = pd.DataFrame()
        for i in range(len(experiments)):
                name_col = "sissim_{}".format(experiments[i])
                df[name_col] = sissim_results[i] + [sum(sissim_results[i])/len(sissim_results[i])]
                name_col = "msssim_{}".format(experiments[i])
                df[name_col] = msssim_results[i] + [sum(msssim_results[i])/len(msssim_results[i])]
                name_col = "psnr_{}".format(experiments[i])
                df[name_col] = psnr_results[i] + [sum(psnr_results[i]) / len(psnr_results[i])]
        df.index = list(range(len(sissim_results[0]))) + ["avg"] # num of images processed
        print(df)
        df.to_excel(output_name +'.xlsx')


if __name__ == '__main__':
        sissim_all = []
        msssim_all = []
        psnr_all = []

        experiments = list(range(1, 17))  # mean k
        #experiments = ['resnext50_32x4d', 'wide_resnet50_2', 'googlenet', 'densenet161', 'inception_v3',
        #               'shufflenet_v2_x1_0', 'mnasnet1_0', 'mobilenet_v2']
        for i, item in enumerate(experiments):
                print("=============== experiment {} ============".format(item))
                sissim, msssim, psnr = test(directory_path='kodak', output_directory_path='kodak_test_results', item = item)
                print("sissim result: ", sissim)
                print("msssim result: ", msssim)
                sissim_all.append(sissim)
                msssim_all.append(msssim)
                psnr_all.append(psnr)
                if i % 2 == 0: # save every 2 experiments
                        output_name = 'semantic_untrained_min_iter_all_k_batch{}'.format(i)
                        save_results(sissim_all, msssim_all, psnr_all, experiments[:i+1], output_name)


        output_name = 'semantic_untrained_min_iter_all_k'
        save_results(sissim_all, msssim_all, psnr_all, experiments, output_name)