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

Our code was run on a 16GB RTX 3080Ti Laptop GPU with 64GB RAM and PyTorch >=1.13, although better GPU/RAM space will allow for faster experimentation.

* Install all requirements required to run the code on a Python >=3.9 environment by:
 ```	
# First, activate a new virtual environment
pip3 install -r requirements.txt
 ```

### Fast Dataset Setup

-  There is a fast, direct mechanism to download and use our datasets implemented in [this repository](https://github.com/hammoudhasan/CLDatasets).
-  Input the directory where the dataset was downloaded into `data_dir` field in `src/opts.py`.
-  All codes in this repository were run on this dataset.

## Recreating the Datasets

- `YOUR_DATA_DIR` would contain two subfolders: `cglm` and `cloc`. Following are instructions to setup each dataset:

### Continual Google Landmarks V2 (CGLM)

#### Download Images

* You can download Continual Google Landmarks V2 dataset by following instructions on their Github repository, run in the `DATA_DIR` directory:
```
wget -c https://raw.githubusercontent.com/cvdfoundation/google-landmark/master/download-dataset.sh
mkdir train && cd train
bash ../download-dataset.sh train 499
```

#### Recreating Metadata

* Download metadata by running the following commands in the `scripts` directory:
```
wget -c https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv
python cglm_scrape.py
```
* Parse the XML files and organize it as a dictionary.
* Ordering used in the paper is available to download [from here]().
* Using the `upload_date` instead of date from EXIF metadata (more unique timestamps and more faithful to the story), we get this [new order file](https://www.robots.ox.ac.uk/~ameya/meta_images_and_authors.pkl). Note: New file generated in June 2023, differs from order file at [CLDatasets](https://github.com/hammoudhasan/CLDatasets) repo. Do not crosscompare.
* However, no substantial changes observed in trends. They remain similar on both, the label correlation does not simply go away (Slightly increases with better ordering, by breaking ties of same timestamps!)
* Now, select only images that are a part of the order file and your dataset should be ready!

### Continual YFCC100M (CLOC)

#### Download Images

* Download the `cloc.txt` file from [this link](https://www.robots.ox.ac.uk/~ameya/cloc.txt) inside the `YOUR_DATASET_DIR/cloc` directory.
* The `cloc.txt` file contains 36.8M image links, sanitizing the original download link from CLOC.
* Download the dataset parallely and scalably using img2dataset, finishes in <a day on a 8-node server (read instructions in `img2dataset` repo for further distributed download options):
```
pip install img2dataset
img2dataset --url_list cyfcc.txt --input_format "txt" --output_form webdataset output_folder images --process_count 16 --thread_count 256 --resize_mode no --skip_reencode True
```

#### Download Meta-data

* [Pending] Then download the order files for [train](https://www.robots.ox.ac.uk/~ameya/cloc_train.txt), [hptune](https://www.robots.ox.ac.uk/~ameya/cloc_hptune.txt) and  [test](https://www.robots.ox.ac.uk/~ameya/cloc_test.txt) to the `YOUR_DATASET_DIR/cloc/` directory.

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
