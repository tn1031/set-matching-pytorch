# set-matching-pytorch
An implementation of "Exchangeable deep neural networks for set-to-set matching and learning" ([link](https://arxiv.org/abs/1910.09972)).

This repository contains two models for set matching problem.

- **Set Transformer**: A model for set-to-element matching problem.
- **Set Matching**: A model for set-to-set matching problem.

## Set up
```bash
git clone https://github.com/tn1031/set-matching-pytorch.git
cd set-matching-pytorch
poetry install
```

## Dataset
Prepare the dataset for "Outfit Compatibility Prediction" tasks. The following two tasks are implemented in this repository.

- Fill in the blank (FITB): The problem of matching incomplete outfit and item. See [3] for details.
- Fill in the N blanks (FINBs): Matching problems between two complementary oufits. See [1] for details.


Go [here](https://anonymity2019.wixsite.com/gp-bpr) and download the IQON3000 dataset. For details on the dataset, please see [2].

### Creating label files
After unzipping, run the following commands to create label files for training and testing.

```
poetry run python make_iqon_dataset.py \
--data_dir       <The path to the directory where the unzipped IQON3000 is located.> \
--min_set_size   <Minimum value of set size. Outfits that are smaller than this will not be used.> \
--n_candidates   <Number of choices in a question (FITB/FINBs).> \
--n_mix          <Mixture number of outfit (FINBs).> \
--max_set_size_x <Query set size (FINBs).> \
--max_set_size_y <Candidate set size (FINBs).> \
--train_size     <Ratio of data for training.> \
--test_size      <Ratio of data for testin. The rest are used for validation.>
```

Four files will be created under the directory specified by `data_dir`.

- train.json, valid.json: for training and validation.
- test_fitb.json: for FITB problem.
- test_finbs.json: for FINBs problem.

## Run
It provides two models, SetTransformer and SetMatching, where SetTransformer is for FITB and SetMatching is for FINBs.

### SetTransformer

```bash
# Training
poetry run python train.py model=set_transformer

# Testing (FITB)
poetry run python eval.py model=set_transformer eval.modelckpt=<path to the set transformer model checkpoint>
```

### SetMatching

```bash
# Training
poetry run python train.py model=set_matching

# Testing (FINBs)
poetry run python eval.py model=set_matching eval.modelckpt=<path to the set matching model checkpoint>
```

### Settings and Hyperparameters
The parameters for each model can be found in [set_transformer.yaml](conf/model/set_transformer.yaml) and [set_matching.yaml](conf/model/set_matching.yaml).
Also, see [config.yaml](conf/config.yaml) for parameters common to training and testing.


## References
- [1] Saito, et al., Exchangeable deep neural networks for set-to-set matching and learning, ECCV (2020).
- [2] Song, et al., GP-BPR: Personalized Compatibility Modeling for Clothing Matching, ACM MM (2019).
- [3] Han, et al., Learning Fashion Compatibility with Bidirectional LSTMs, ACM MM (2017).
