### FinRL Transformer Models ###

### Description ### 
The objective is to conduct an empirical study into the training and performance of transformer models under different loss functions. We leverage Yahoo Finance, Pytorch, and various machine learning modules.

### Goals ###
* Assess the effectiveness of employing Mean Squared Error and Mean Absolute Error as loss functions in transformer models.
* Evaluate the impact of Cross-Entropy Loss on transformers, for time series predictions.
* Contrast and compare results processed from Long Short-Term Memory and Transformers.
* Establish a robust baseline model as a basis for FinRL’s reinforcement models.

### Stack ###
PyTorch 
Python

### Members - under the guidance of Prof. Yanglet Xiao-Yang Liu ### 
Yun Zhe Chen (Project Lead)
cheny73@rpi.edu
David C
chongd@rpi.edu
Wenjie Chen
chenw20@rpi.edu
Andy Zhu
zhua6@rpi.edu
Derrick L
lind7@rpi.edu
Hongwei L
lih40@rpi.edu


### Milestones ###
## Project Initialization & Planning ##
* Gather up resources
* Review published research 
* Review relevant machine learning topics and become familiar with Pytorch 
## Study Phase ## 
* Learning ARMA + Regression LTSM model + Transformer for practice
# Data Pipelining & Collection # 
* Collect and preprocess data from Yahoo Finance 
* Compose a time series for collected data 
# Testing Long Short-Term Memory Model #  
* Apply standard LSTM model training using PyTorch
* Implement and test the LSTM model by employing Mean Squared Error and Mean Absolute Error as loss functions 
* Report current progress for discussion with professor
# Testing Transformer Model #  
* Apply standard transformer model training using PyTorch 
* Implement and test the transformer model by employing Mean Squared Error and Mean Absolute Error as loss functions
# Cross-Entropy Loss Functions Evaluation # 
* Implement cross-entropy to test the transformer model as a loss function
* Analyze results
# Finalize Findings & Interpretation #
* Conduct compare and contrast for LSTM and transformer models 
* Prepare poster for presentation 

Project Link  https://github.com/blitzionic/FinRL---Stock-Prediction

Resoures: 
* https://github.com/AI4Finance-Foundation/FinRL
* [Yahoo Downloader using yfinance to fetch data from Yahoo Finance](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/preprocessor/yahoodownloader.py)
