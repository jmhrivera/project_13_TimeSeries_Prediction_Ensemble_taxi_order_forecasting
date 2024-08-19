# Project 13: Time Series Prediction Ensemble for Taxi Order Forecasting

## Overview
Sweet Lift Taxi has collected historical data on taxi orders at airports. To attract more drivers during peak hours, we need to predict the number of taxi orders for the next hour. The goal is to build a model to make this prediction and ensure the RMSE on the test set does not exceed 48.

## Instructions
1. Download the data and resample it by one hour.
2. Analyze the data.
3. Train different models with varying hyperparameters. The test sample should be 10% of the initial dataset.
4. Test the models using the test sample and provide a conclusion.

## Dataset Description
The data is stored in the file `taxi.csv`. The number of orders is in the column `num_orders`.

## Setup
1. Clone the repository:
```sh
git clone https://github.com/your_username/project_17_TimeSeries_Prediction_Ensemble_taxi_order_forecasting.git
cd project_17_TimeSeries_Prediction_Ensemble_taxi_order_forecasting
```

2. Install the required packages:
```sh
pip install -r requirements.txt
```
