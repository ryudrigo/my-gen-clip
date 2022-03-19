# my-gen-clip

In-progress implementation of GEN-CLIP

## Installation

Create a conda enviroment based on environment.yaml file

`cd taming-transformers`

`conda env create --file environment.yml`

`conda activate vqgen`

## Usage

The basic usage goes like this (futher instructions in the taming-transformers repo README)

`python main.py --base configs/ffhq_thumbnails_transformer.yaml -t True --gpus 0,`

The config file I used to train the first stage is configs/custom_vqgan.yaml For the second stage, I used configs/ffhq_thumbnails_transformer.yaml

Please note that training with those configurations requires the ffhq-thumbnails dataset, which is not included in this repo. You can use other datasets, further instructions can be found at 

https://github.com/CompVis/taming-transformers#training-on-custom-data
