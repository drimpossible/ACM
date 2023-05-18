# ACM (Under construction, will be ready before May end)

This repository contains the code for the paper:

**Online Continual Learning Without the Storage Constraint**  
[Ameya Prabhu](https://drimpossible.github.io), [Zhipeng Cai](), [Puneet Dokania](https://puneetkdokania.github.io), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Vladlen Koltun](), [Ozan Sener]()
[[Arxiv](https://arxiv.org/)]
[[PDF]()]
[[Bibtex](https://github.com/drimpossible/ACM/#citation)]

## Installation and Dependencies

* Install all requirements required to run the code on a Python 3.9 environment by:
 ```	
# First, activate a new virtual environment
pip3 install -r requirements.txt
 ```
 
* Create three additional folders in the repository `data/`, `data_scripts/` and `logs/` which will store the datasets and logs of experiments. Point `--order_file_dir` and `--log_dir` in `src/opts.py` to locations of these folders.

## Generating Continual Google Landmarks V2 Dataset

* You can download Continual Google Landmarks V2 metadata from [this link]() directly. To reproduce this follow the below instructions:
```
cd data_scripts/
wget https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv
wget https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv
cd ../scripts/
python scrape_flickr.py
```

## Generating Continual YFCC100M (CLOC) Dataset

* 

## Usage

## Replication

## Additional Experiments

* To reproduce our KNN scaling graphs (Figure 1b), please run the following on a computer with high RAM:
```
cd scripts/
python knn_scaling.py
python plot_knn_results.py
```


##### If you discover any bugs in the code please contact me, I will cross-check them with my nightmares.

## Citation

We hope ACM is a strong method for comparison, and this idea/codebase is useful for your cool CL idea! To cite our work:

```
@article{prabhu2023online,
  title={Online Continual Learning Without the Storage Constraint},
  author={Prabhu, Ameya and Cai, Zhipeng and Dokania, Puneet and Torr, Philip and Koltun, Vladlen and Sener, Ozan},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```
