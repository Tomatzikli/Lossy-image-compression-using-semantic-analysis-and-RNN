import pandas as pd
import torch
from test import test
import os


def save_results(msssim_results, result_sizes, orig_sizes, output_name):
        df = pd.DataFrame()
        df["orig_sizes"] = orig_sizes + [""]
        for i in range(len(experiments)):
          msssim_results[i] += [sum(msssim_results[i])/len(msssim_results[i])]
          name_col = "msssim_{}".format(experiments[i])
          df[name_col] = msssim_results[i]
          name_col = "result_sizes_{}".format(experiments[i])
          df[name_col] = result_sizes[i] + [""]
          sizes_ratio = torch.tensor(orig_sizes) / torch.tensor(result_sizes[i])
          sizes_ratio = torch.cat((sizes_ratio, torch.tensor([sum(sizes_ratio) / len(sizes_ratio)])))
          name_col = "sizes_ratio_{}".format(experiments[i])
          df[name_col] = sizes_ratio

        df.index = list(range(len(msssim_results[0])-1)) + ["avg"] # num of images processed

        print(df)
        df.to_excel(output_name +'.xlsx')


## main
msssim_all = []
result_sizes_all = []
experiments = list(range(1,25)) # mean k

for item in experiments:
  print("=============== experiment {} ============".format(item))
  msssim, result_sizes = test(directory_path='kodak', output_directory_path='kodak_test_results', item=item)
  msssim_all.append(msssim)
  result_sizes_all.append(result_sizes)

output_name = 'test_only_rnn'

orig_sizes = []
for i in range(1,25):
  if i <= 9:
    input_path = "kodak/kodim0{}.png".format(i)
  else:
    input_path = "kodak/kodim{}.png".format(i)
  orig_sizes.append(os.path.getsize(input_path)//1024)

save_results(msssim_all, result_sizes_all, orig_sizes, output_name)




