#  An Automated Portfolio Trading System with Feature

## Objective 

Inspired by the article entitled [An Automated Portfolio Trading System with Feature Preprocessing and Recurrent Reinforcement Learning](https://paperswithcode.com/paper/an-automated-portfolio-trading-system-with) written by Lin Li, we aim at implementing a fully automated trading system which incorporates a portfolio weight rebalance function and handles multiple assets. The trading bot is based on recurrent reinforcement learning and is developed in `python`. 

## Conceptual framework

The below schema briefly depicts the two main parts of the trading bot namely the data preprocessing layers and the recurrent reinforcement learning (RRL) model. The following sections give more details about each step used to build the bot, as well as the results obtained during backtests.

![](imgs/rrl-pca-dwt.png)

## Data 

In this part, we explain the data source used to realize the project and the preprocessing steps that are implemented to remove the noise in the raw data and uncover the general pattern underlying the financial data set.

### Configuration

#### Yahoo Finance data

Since the trading system is supposed to run continuously on daily data, the [`yfinance`](https://pypi.org/project/yfinance/) library is useful to retrieve accurate financial data on multiple stocks. It is an open-source tool that uses Yahoo's publicly available APIs, and is intended for research and educational purposes.

#### Assets

As in the Lin Li's article, we used the 8 subsequent financial assets as input in the RRL trading system. These stocks are listed in the S&P500 index which is representative of the general stock market condition in
the US. When downloading the data from Yahoo Finance, Open, High, Low, Close and Volume are returned for each stock. The study is realised between 2009/12/31 and 2017/12/29. 

|  Ticker | Company  |
|---|---|
| XOM | Exxon Mobil Corporation |
| VZ | Verizon Communications Inc. |
| NKE | Nike, Inc. |
| AMAT | Applied Materials, Inc. |
| MCD | McDonald's Corporation |
| MSFT | Microsoft Corporation |
| AAP | Advance Auto Parts, Inc. |
| NOV | Nov, Inc. |

#### Technical indicators

Technical indicators are heuristic or pattern-based signals produced by the price, volume, and/or open interest of a security or contract used by traders who follow technical analysis. In other words, they summarize the general pattern of the time series. While 4 groups of technical indicators are mentioned in the article, we solely use 3 types as depicted in the following table.

|  Momentum | Volatility  | Volume |
|---|---|---|
| Momentum (MOM)  | Average True Range (ATR) | Chaikin Oscillator (CO) |
| Moving Average Convergence Divergence (MACD) | Normalized Average True Range (NATR) | On Balance Volume (OBV) |
| Money Flow Index (MFI) | | |
| Relative Strength Index (RSI) | | |

Both the [`ta`](https://pypi.org/project/ta/) and [`TA-Lib`](https://mrjbq7.github.io/ta-lib/) Python libraries are leveraged to compute the indicators without much difficulty. 

We note <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/6937e14ec122765a9d014f2cbcf4fcfe.svg?invert_in_darkmode" align=middle width=13.13115539999999pt height=22.465723500000017pt/> the set of technical indicators such that <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/f0d67bdf237663ea1f639eb508c9b9de.svg?invert_in_darkmode" align=middle width=52.400435999999985pt height=24.65753399999998pt/>.

#### Normalization 

To avoid scaling issues, each technical indicator feature is normalized using the z-score: 

<p align="center"><img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/b1517af1942bc2bb613fe344028fb9f5.svg?invert_in_darkmode" align=middle width=93.3193173pt height=38.350841100000004pt/></p>

where <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/05330d44fdfc78bc5a122fa403597c61.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=26.97711060000001pt/> is the mean and <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/741d09eb889c94b1dcb67eaf030e1788.svg?invert_in_darkmode" align=middle width=19.380205349999994pt height=14.15524440000002pt/> the standard deviation. 

### Dimension reduction and signal processing

One of the main stake that arise when training a machine learning model, is the agent's ability to generalize on unseen data. In other words the ML agent needs to learn the general pattern of the data, and noise has to be removed. 

#### Principal Component Analysis (PCA)

PCA is the first technique used in the preprocessing layer and aims at reducing the dimension of the input data. To that end, PCA identifies principal axes that represent the directions of maximum variance of the input. In our project, the normalized indicators in <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/6937e14ec122765a9d014f2cbcf4fcfe.svg?invert_in_darkmode" align=middle width=13.13115539999999pt height=22.465723500000017pt/> are decomposed by PCA such that the sum of the variance explained by principal components explains at least 95% of the total variance. We thus obtain a new set of features <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/6d490ad7efc6a5597550a2981b0832cb.svg?invert_in_darkmode" align=middle width=52.791809399999984pt height=24.7161288pt/>, with <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/ae4aaeb3f0fe9130e5184fa93e981578.svg?invert_in_darkmode" align=middle width=57.01231139999999pt height=24.7161288pt/>. The [`sklearn`](https://scikit-learn.org/stable/) library is used to implement PCA.

#### Discrete Wavelet Transform (DWT)

ALthough PCA is a powerful technique for dimension reduction, some local noise may persist in the training data. Consequently, the DWT method is applied on the principal components in <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/c2e01307da54ba55d6a513859375c2a0.svg?invert_in_darkmode" align=middle width=16.921109699999988pt height=24.7161288pt/>. First, the input data is decomposed into several coefficients so as to separate the general trend of the signal from the local noise. Then, we apply soft thresolding technique on the coefficients. Finally, the denoised version of the original signal is obtained with the inverse DWT method. The [`PyWavelets `](https://pywavelets.readthedocs.io/en/latest/) is used to implement Discrete Wavelet Transform. 

### Train / trading split 

As shown by the conceptual schema, the data is divided into training and trading (validation) batches of length <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/36786a3fa8563a77d5ff9dbfec5342a4.svg?invert_in_darkmode" align=middle width=58.46457044999998pt height=22.465723500000017pt/> days which are defined as folllows: 

<p align="center"><img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/ef4020ab0409ab4be09994f2ffbb488e.svg?invert_in_darkmode" align=middle width=162.4745595pt height=81.57117869999999pt/></p>

where <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/37ab1198f7150fec26074382d942efd6.svg?invert_in_darkmode" align=middle width=19.399587899999993pt height=22.465723500000017pt/> is the transformed feature matrix and <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/4f00e31c4c3e78d512ef4de2e94e0acd.svg?invert_in_darkmode" align=middle width=14.456681249999992pt height=14.15524440000002pt/> the target variable vector. 

It is relevant to note that normalization and PCA are only fitted on the training batches and the technical indicators are calculated on each batch separately. The idea behind this is to ensure that the model's performance is an accurate reflection of its ability to generalize to new data.  

## The RRL model

Once the data fully prepreocessed and the training and trading batches created, the recurrent reinforcement lerning model can start its learning process. 

### Objective

Based on the preprocessed technical indicators, the RRL agent aims at rebalancing the portfolio which is composed of <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/0e51a2dede42189d77627c4d742822c3.svg?invert_in_darkmode" align=middle width=14.433101099999991pt height=14.15524440000002pt/> assets with corresponding weights, denoted <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/0be8cf21cef33ebcd1916a11a502bc51.svg?invert_in_darkmode" align=middle width=153.13712864999997pt height=24.7161288pt/>. 

<img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/2e17162e64120b6646da8e66ff732951.svg?invert_in_darkmode" align=middle width=18.550210799999995pt height=23.378942399999996pt/> is updated at each period with a view to maximize Sharpe ratio defined as: 

<p align="center"><img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/526355293583d260b2ef8be95c03f8d3.svg?invert_in_darkmode" align=middle width=307.52253509999997pt height=69.58589715pt/></p>

Given <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/d7eb8872131c8836aeeffa6f5e835cec.svg?invert_in_darkmode" align=middle width=13.569240299999997pt height=15.068463299999998pt/> the vector of assets' returns, <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode" align=middle width=7.928075099999989pt height=22.831056599999986pt/> the transaction fees and <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/354ce827d4413f6dde43356a2cb88096.svg?invert_in_darkmode" align=middle width=109.0693065pt height=24.7161288pt/>, the portfolio return at time <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> is:

<p align="center"><img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/81fdda1de09d7ac66fc4c12a8db55e02.svg?invert_in_darkmode" align=middle width=335.69642699999997pt height=18.6136995pt/></p>

Note we use positions computed at time <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/d6386b3a3242cb5aef731c98835d14fd.svg?invert_in_darkmode" align=middle width=34.24649744999999pt height=21.18721440000001pt/> to obtain returns at time <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/4f4f4e395762a3af4575de74c019ebb5.svg?invert_in_darkmode" align=middle width=5.936097749999991pt height=20.221802699999984pt/> since there is a usual 2-day delay when implementing daily trading strategies in practice. In the case where the positions are the same from time <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/ad2cc37c150cf9deb634941c3b981e11.svg?invert_in_darkmode" align=middle width=34.24649744999999pt height=21.18721440000001pt/> to time <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/d6386b3a3242cb5aef731c98835d14fd.svg?invert_in_darkmode" align=middle width=34.24649744999999pt height=21.18721440000001pt/>, the <img src="https://rawgit.com/PeDiot/rrl-trading-system/master/svgs/38f1e2a089e53d5c990a82f284948953.svg?invert_in_darkmode" align=middle width=7.928075099999989pt height=22.831056599999986pt/> term disappears from the formula.

### Architecture

![](imgs/rrl.png)

### Training 

### Validation / Trading

## Backtest

![](imgs/cum-profits-rrl-pca-dwt-3.png)