# An End-to-End Deep Learning Image Compression Framework Based on Semantic Analysis
https://doi.org/10.3390/app9173580

# encode.py --input_path kodak/kodim23.png --codes_output_directory compressed_codes --model checkpoint/encoder_epoch_00000025.pth
This will output binary codes saved in .npz format and a zipped pickle.

decode.py --codes_input_directory compressed_codes --output_directory compressed_output  --model checkpoint/encoder_epoch_00000025.pth
The only required argument is input_path, the rest has a default value. 
This will output the reconstructed image.

# Original:
![kodim18](https://user-images.githubusercontent.com/59319073/132473032-dd46e455-8cb2-435f-9637-af553b65e530.png)


#Decoded bpp 0.75:
# ![kodak_17_k6](https://user-images.githubusercontent.com/59319073/132472816-eecc0107-bff3-4ea8-ad84-b770fe56dd05.jpg)

