# Data

Raw data is not committed to this repo (too large, and it's a public Kaggle
dataset anyone can re-download).

Source: Corporación Favorita Grocery Sales Forecasting (Kaggle)
https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting

To reproduce:
1. Create a free Kaggle account and generate an API token
   (Kaggle account settings → Create New Token → downloads kaggle.json)
2. Place kaggle.json in ~/.kaggle/kaggle.json
3. Run: kaggle competitions download -c favorita-grocery-sales-forecasting -p data/raw
4. Unzip into data/raw/
