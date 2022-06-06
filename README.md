# ACM

Codebase for adaptive continual memory

## Installation and Dependencies

* Install all requirements required to run the code on a Python 3.9 environment by:
 ```	
# First, activate a new virtual environment
pip3 install -r requirements.txt
 ```
 
* Create three additional folders in the repository `data/`, `data_scripts/` and `logs/` which will store the datasets and logs of experiments. Point `--order_file_dir` and `--log_dir` in `src/opts.py` to locations of these folders.

## Generating GLDv2 Metadata

* You can download GLDv2 preprocessed metadata from [this link]() directly. However, to reproduce this follow the below instructions:
```
cd data_scripts/
wget https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv
wget https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv
cd ../scripts/
python scrape_flickr.py
```
