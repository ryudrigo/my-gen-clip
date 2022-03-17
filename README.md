# my-gen-clip

In-progress implementation of GEN-CLIP

## Installation

Create a conda enviroment based on environment.yaml file
(Currently it complains about CLIP version, because CLIP is installed directly from their github. So what I do is creating the conda environment, installing CLIP from their github and, then, updating the environment so it installs the other dependencies)

`cd taming-transformers`

`conda env create --file environment.yml`

`conda activate vqgen`

`pip install git+https://github.com/openai/CLIP.git`

`conda env update --file local.yml`

(Yes, this is very contrived. Yes, fixing it is on the to-do list)

## Usage

The basic usage goes like this (futher instructions in the taming-transformers README)

`python main.py --base configs/ffhq_thumbnails_transformer.yaml -t True --gpus 0,`

The config file I used to train the first stage is configs/custom_vqgan.yaml For the second stage, I used configs/ffhq_thumbnails_transformer.yaml

Please note that training with those configurations requires the ffhq-thumbnails dataset, which is not included in this repo. You can use other datasets, further instructions can be found at 

https://github.com/CompVis/taming-transformers#training-on-custom-data

## Understanding

Relevant differences to the original taming-transformers are mostly on the file taming-transformers/taming/models/clip_transformer.py
