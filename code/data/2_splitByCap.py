import path
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

# Reading csv-file using a relative path, based on the folder structure of the github project
file_path = path.Path(__file__).parent / "scrapedData.csv"
with file_path.open() as dataset_file:
    df_scraped = pd.read_csv(dataset_file)
    df_scraped.dropna(inplace=True)
    df_scraped.reset_index(inplace=True, drop=True)

def get_large_cap(df):
    large_query = np.where((df["MarketCap"] >= 6000000.0) & (df["Year"] >= 2012))
    return df.loc[large_query]

def get_small_cap(df):
    large_query = np.where((df["MarketCap"] <= 3000000.0) & (df["Year"] >= 2012))
    return df.loc[large_query]

def get_target(stock_return, median):
    if stock_return >= median:
        return 1
    else:
        return 0

def calculate_target(df_ticker_data):
    data_frame_output = pd.DataFrame()
    dates = list(df_ticker_data.Date.unique())
    dates.sort()
    for date in dates:
        sub_date_data = df_ticker_data[df_ticker_data.Date == date]
        returns_median = sub_date_data['Returns'].median()
        sub_date_data['Median'] = returns_median
        sub_date_data['Target'] = sub_date_data['Returns'].apply(get_target, args=(returns_median,))
        data_frame_output = pd.concat([data_frame_output, sub_date_data], ignore_index=False)
    return data_frame_output

df_large = get_large_cap(df_scraped)
df_small = get_small_cap(df_scraped)

print("Calculating median and adding median + target to each row on large cap output")
df_large_output = calculate_target(df_large)

print("Calculating median and adding median + target to each row on small cap output")
df_small_output = calculate_target(df_small)

# save to csv
df_large_output.to_csv('df_large_output.csv')
df_small_output.to_csv('df_small_output.csv')
