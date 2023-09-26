# MAGNN

[![Build Status](https://travis-ci.org/RaRe-Technologies/gensim.svg?branch=develop)](https://travis-ci.org/RaRe-Technologies/gensim)
[![GitHub release](https://img.shields.io/github/release/rare-technologies/gensim.svg?maxAge=3600)](https://github.com/RaRe-Technologies/gensim/releases)
[![Wheel](https://img.shields.io/pypi/wheel/gensim.svg)](https://pypi.python.org/pypi/gensim)
[![DOI](https://zenodo.org/badge/DOI/10.13140/2.1.2393.1847.svg)](https://doi.org/10.13140/2.1.2393.1847)
[![Mailing List](https://img.shields.io/badge/-Mailing%20List-brightgreen.svg)](http：//www.seek-data.com)

Financial time series forecasting with multi-modality graph neural network


## Usage

1. Prepare your data

   * We have retained the data used as a sample in the ```event```, ```news```, and ```price``` subfolders in the ```data``` folder and stored it in the form of ```.pkl```. These files can be opened using the ```np.load() ```function to view their specific styles.
   * At the same time, the sample relationship diagram between companies is placed in the ```example_company_relation.pkl ``` under the``` dataset``` folder. 

2. Setup your env

   * We recommend you to create a new python environment for MAGNN. You might run this command in your anaconda prompt in order to create a new environment:

     ``` 
     conda create -n magnn python==3.6.13
     ```

     We recommend to use Python 3.6 in our model. 

   * After enter the magnn folder, you might install required packages using command below:

     ``` 
     pip install -r requirements.txt
     ```

3. Run MAGNN

* All functions are integrated in magnn.py, you only need to execute ```python magnn.py``` in your virtual environment to run. The results of the operation will be placed in ``` magnn_result.csv```. You might use ```pandas``` to reveal it.
* If you want to load your own data to MAGNN, do not forget to change train/test period tuples which are defined in ```./dataset/constant.py```

## Project Description
- **dataset**  
   This module contains helper functions to initiate train/valid/test datasets including price data, stock event and stock news data for our models.
   
- **model**  
   This module contains all our models(magnn, price-lstm, event-embedding, news-embedding).
   
- **tools**
	This module contains some simple implementation versions of some non-open source functions. Readers can modify the functions as needed.

## Citing

* If you find **MAGNN** is useful for your research, please consider citing the following papers:

    ``` latex
    @inproceedings{han2023efficient,
        title={Financial time series forecasting with multi-modality graph neural network},
        author={Dawei Cheng, Fangzhou Yang, Sheng Xiang, Jin Liu},
        booktitle={Pattern Recognition},
        year={2022},
        paper website={https://www.sciencedirect.com/science/article/abs/pii/S003132032100399X},
      }



