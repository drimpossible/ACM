# ACM

This repository contains the code for the paper:

**Online Continual Learning Without the Storage Constraint**  
[Ameya Prabhu](https://drimpossible.github.io), [Zhipeng Cai](https://zhipengcai.github.io/), [Puneet Dokania](https://puneetkdokania.github.io), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Vladlen Koltun](https://vladlen.info/), [Ozan Sener](https://ozansener.net/)
[[Arxiv](https://arxiv.org/abs/2305.09253)]
[[PDF](https://drimpossible.github.io/documents/ACM.pdf)]
[[Bibtex](https://github.com/drimpossible/ACM/#citation)]

<p align="center">
  <img src="https://github.com/drimpossible/ACM/blob/main/Model.png" width="600" alt="Figure which describes our ACM model">
</p>

## Installation and Dependencies

* Install all requirements required to run the code on a Python 3.9 environment by:
 ```	
# First, activate a new virtual environment
pip3 install -r requirements.txt
 ```
 
* Create three additional folders in the repository `data/`, `data_scripts/` and `logs/` which will store the datasets and logs of experiments. Point `--order_file_dir` and `--log_dir` in `src/opts.py` to locations of these folders.

## Dataset Setup

- `YOUR_DATA_DIR` will contain two subfolders: `cglm` and `cloc`. Following are instructions to setup each dataset:

### Continual Google Landmarks V2 (CGLM)

* You can download Continual Google Landmarks V2 metadata from [this link]() directly. To reproduce this follow the below instructions:
```
cd data_scripts/
wget https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv
wget https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv
cd ../scripts/
python scrape_flickr.py
```

### Continual YFCC100M (CLOC)

* Download the `cloc.txt` file from [this link](https://www.robots.ox.ac.uk/~ameya/cloc.txt) inside the `YOUR_DATASET_DIR/cloc` directory.
* Download the dataset parallely and scalably using img2dataset (read instructions in `img2dataset` repo for further distributed download options):
```
pip install img2dataset
img2dataset --url_list cyfcc.txt --input_format "txt" --output_form webdataset output_folder images --process_count 16 --thread_count 256 --resize_mode no --skip_reencode True
```
* Then download the order files for [train](https://www.robots.ox.ac.uk/~ameya/cloc_train.txt), [hptune](https://www.robots.ox.ac.uk/~ameya/cloc_hptune.txt) and  [test](https://www.robots.ox.ac.uk/~ameya/cloc_test.txt) to the `YOUR_DATASET_DIR/cloc/` directory.

### Alternative Fast Dataset Setup

-  There is a fast, direct mechanism to download and use our datasets implemented in [this repository](https://github.com/hammoudhasan/CLDatasets).
-  Input the directory where the dataset was downloaded into `data_dir` field in `src/opts.py`.


## Running the Code

## Replication

## Additional Experiments

* To reproduce our KNN scaling graphs (Figure 1b), please run the following on a computer with high RAM:
```
cd scripts/
python knn_scaling.py
python plot_knn_results.py
```


* To reproduce the blind classifier, please run the following:
```
cd scripts/
python run_blind.py
```


##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.

## Citation

We hope ACM is a strong method for comparison, and this idea/codebase is useful for your cool CL idea! To cite our work:

```
@article{prabhu2023online,
  title={Online Continual Learning Without the Storage Constraint},
  author={Prabhu, Ameya and Cai, Zhipeng and Dokania, Puneet and Torr, Philip and Koltun, Vladlen and Sener, Ozan},
  journal={arXiv preprint arXiv:2305.09253},
  year={2023}
}
```
