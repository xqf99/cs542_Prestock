# cs542_Prestock
The goal of this project is to determine anomalies in Bitcoin price changes, and auto-generate a news article reporting on that anomaly.
We implemented 2 models, found in the Bitcoin_Predict and News_worthy_sequence_pre folder.

Model 1:
For this model, you will need to have installed the following dependencies: pandas, quandl, numpy, tensorflow, and matplotlib.
In Bitcoin_Predict, run all the cells in Pre_bitcoin.ipynb.

Model 2: 
For this model, you will need to have installed the following dependencies: scipy, matplotlib, numpy, pandas, pandas_datareader, mpld3, sklearn, and keras.
In Newsworthy_sequence_pre, run all the cells in the following files in the same order: Data_PreProcessing.ipynb, Model.ipynb, Predict_Model.ipynb.
In Predict_Model.ipynb, you will also be prompted to enter a targeted time period, and our model will return articles for newsworthy events about Bitcoin price changes in that period.

