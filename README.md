# Lossy-image-compression-using-semantic-analysis-and-RNN
https://doi.org/10.3390/app9173580

encode.py --input_path kodak/kodim23.png --codes_output_directory compressed_codes --model checkpoint/encoder_epoch_00000025.pth
This will output binary codes saved in .npz format and a zipped pickle.

decode.py --codes_input_directory compressed_codes --output_directory compressed_output  --model checkpoint/encoder_epoch_00000025.pth
The only required argument is input_path, the rest has a default value. 
This will output the reconstructed image.
