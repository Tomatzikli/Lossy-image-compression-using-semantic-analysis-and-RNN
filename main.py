import numpy as np
from test import test
import pandas as pd
import os
import torch


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

        orig_sizes = []
        for i in range(1, 25):
                if i <= 9:
                        input_path = "kodak/kodim0{}.png".format(i)
                else:
                        input_path = "kodak/kodim{}.png".format(i)
                orig_sizes.append(os.path.getsize(input_path) // 1024)

        # experiments = list(range(1,25)) # mean k
        experiments = list(range(1, 13))  # mean k
        #experiments = ['resnext50_32x4d', 'wide_resnet50_2', 'googlenet', 'densenet161', 'inception_v3',
        #               'shufflenet_v2_x1_0', 'mnasnet1_0', 'mobilenet_v2']
        for i, item in enumerate(experiments):
                print("=============== experiment {} ============".format(item))
                sissim, msssim,  result_sizes = test(directory_path='kodak', output_directory_path='kodak_test_results', item = item)
                print("sissim result: ", sissim)
                print("msssim result: ", msssim)
                sissim_all.append(sissim)
                msssim_all.append(msssim)
                result_sizes_all.append(result_sizes)
                if i % 2 == 0: # save every 2 experiments
                        output_name = 'semantic_trained__min_iter_all_k_batch{}'.format(i)
                        save_results(sissim_all, msssim_all, result_sizes_all, orig_sizes, experiments, output_name)


        output_name = 'semantic_trained__min_iter_all_k'
        save_results(sissim_all, msssim_all, result_sizes_all, orig_sizes, experiments, output_name)