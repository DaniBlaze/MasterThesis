import path
import pandas as pd

# Reading csv-file using a relative path, based on the folder structure of the github project
file_path = path.Path(__file__).parent / "df_large_output.csv"
with file_path.open() as dataset_file:
    df_large_input = pd.read_csv(dataset_file)

file_path = path.Path(__file__).parent / "df_small_output.csv"
with file_path.open() as dataset_file:
    df_small_input = pd.read_csv(dataset_file)

def normalize_data(returns,mean,std):
    return (returns-mean)/std

def generate_normalized_study_period(data_input, file_prefix):
    data_input = data_input.sort_values(['Date'])
    data_input = data_input.reset_index(drop = True)
    unique_dates = list(data_input.Date.unique())

    years = 9
    for i in range(years):
        study_date = unique_dates[250*i : 250*(i+4)]        
        study_period = data_input[data_input.Date.isin(study_date)]
        
        train_date = unique_dates[250*i : 250*(i+3)]
        train_data = data_input[data_input.Date.isin(train_date)]
        
        trade_date = unique_dates[250*(i+3) : 250*(i+4)]
        trade_data = data_input[data_input.Date.isin(trade_date)]

        train_returns_mean = train_data.Returns.mean()
        train_returns_std = train_data.Returns.std()   
        study_period['Normalized_Returns'] = study_period['Returns'].apply(normalize_data,args=(train_returns_mean,train_returns_std))

        study_period.to_csv('study_period/'+str(file_prefix)+'_study_period_'+str(i)+'.csv')

generate_normalized_study_period(df_large_input, 'large')
generate_normalized_study_period(df_small_input, 'small')