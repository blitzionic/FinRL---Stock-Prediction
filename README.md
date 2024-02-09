### FinRL Transformer Models ###  
Spring 2024 Project Proposal

### Description ### 
The objective of this project is to conduct an empirical study into the training performance of transformer models in the context of different machine learning loss functions starting with a stock prediction model from Yahoo Finance data mainly using Pytorch.

### Goals ###
Assess the effectiveness of employing Mean Squared Error and Mean Absolute Error as loss functions in transformers in Large Language Models.
Evaluate the impact of Cross-Entropy Loss on transformers, for time series predictions.
Contrast and compare results processed from Long Short-Term Memory and Transformers.
Establish a robust baseline model as a basis for FinAL’s reinforcement models.

### Stack ###
PyTorch 
Python
R

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
Project Initialization & Planning
Gather up materials (ALL)
Published researches 
Previous relevant project 
Review necessary machine learning topics (ALL)
Learning Phase (Mid-Feb)
Learning ARMA + Regression LTSM model + Transformer for practice (ALL)
Data Pipelining & Collection (End of Fed)
Collect and preprocess data from Yahoo Finance (Andy, Yun Zhe)
Compose a time series for collected data (Hongwei, David)
Testing Long Short-Term Memory Model (Mid-March) 
Apply standard LSTM model training using PyTorch (Yun Zhe)
Implement and test the LSTM model by employing Mean Squared Error and Mean Absolute Error as loss functions (Hongwei, David, Derrick)
Report current progress for discussion (Anytime with new findings)
Testing Transformer Model (Mid-April) 
Apply standard transformer model training using PyTorch (Yun Zhe, Andy)
Implement and test the transformer model by employing Mean Squared Error and Mean Absolute Error as loss functions (Andy, Yun Zhe)
Cross-Entropy Loss Functions Evaluation (End of March)
Implement cross-entropy to test the transformer model as a loss function (Wenjie, Yun Zhe)
Analyze results (Wenjie)
Finalize Findings & Interpretation (End of April) 
Conduct compare and contrast for LSTM and transformer models (Yun Zhe, Andy)
Compose a written report on the evaluation (ALL)
Organize results into Google Slides for presentation (ALL)

Project Link  https://github.com/blitzionic/FinRL---Stock-Prediction


[Yahoo Downloader using yfinance to fetch data from Yahoo Finance](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/preprocessor/yahoodownloader.py)
