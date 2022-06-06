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

* You can download GLDv2 preprocessed metadata from [this link]() directly. To reproduce this follow the below instructions:
```
cd data_scripts/
wget https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv
wget https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv
cd ../scripts/
python scrape_flickr.py
```

* You can download the YFCC100M preprocessed metadata and images by following [this link]() directly. To reproduce this, please contact <zhipeng.cai>.

## Usage

## Replication

## Additional Experiments

* To reproduce our KNN scaling graphs (Figure 2), please run the following on a computer with high RAM:
```
cd scripts/
python knn_scaling.py
python plot_knn_results.py
```

* To reproduce CLOC results from our paper, please run the following:
```
Ask Zhipeng to fill this in
```

##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.

## Citation

We hope ACM is a strong method for comparison, and this idea/codebase is useful for your cool CL idea! To cite our work:

```
@inproceedings{prabhu2020online,
  title={Online Continual Learning Without the Storage Constraint},
  author={Prabhu, Ameya and Cai, Zhipeng and Dokania, Puneet and Torr, Philip and Koltun, Vladlen and Sener, Ozan},
  booktitle={TBA},
  month={TBA},
  year={TBA}
}
```