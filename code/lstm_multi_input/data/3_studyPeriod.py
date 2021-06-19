import path
import pandas as pd

# Reading csv-file using a relative path, based on the folder structure of the github project
file_path = path.Path(__file__).parent / "df_large_output.csv"
with file_path.open() as dataset_file:
    df_large_input = pd.read_csv(dataset_file)

file_path = path.Path(__file__).parent / "df_small_output.csv"
with file_path.open() as dataset_file:
    df_small_input = pd.read_csv(dataset_file)

def normalize_data(input,mean,std):
    return (input-mean)/std

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

        train_volume_mean = train_data.Volume.mean()
        train_volume_std = train_data.Volume.std()   
        study_period['Normalized_Volume'] = study_period['Volume'].apply(normalize_data,args=(train_volume_mean,train_volume_std))

        train_vix_mean = train_data['^VIX'].mean()
        train_vix_std = train_data['^VIX'].std()   
        study_period['Normalized_VIX'] = study_period['^VIX'].apply(normalize_data,args=(train_vix_mean,train_vix_std))

        train_bz_mean = train_data['BZ=F'].mean()
        train_bz_std = train_data['BZ=F'].std()   
        study_period['Normalized_BZ=F'] = study_period['BZ=F'].apply(normalize_data,args=(train_bz_mean,train_bz_std))

        train_tnx_mean = train_data['^TNX'].mean()
        train_tnx_std = train_data['^TNX'].std()   
        study_period['Normalized_TNX'] = study_period['^TNX'].apply(normalize_data,args=(train_tnx_mean,train_tnx_std))

        train_nok_mean = train_data['NOK=X'].mean()
        train_nok_std = train_data['NOK=X'].std()   
        study_period['Normalized_NOK=X'] = study_period['NOK=X'].apply(normalize_data,args=(train_nok_mean,train_nok_std))

        train_avg50_mean = train_data['avg_50'].mean()
        train_avg50_std = train_data['avg_50'].std()   
        study_period['Normalized_Avg_50'] = study_period['avg_50'].apply(normalize_data,args=(train_avg50_mean,train_avg50_std))

        train_avg200_mean = train_data['avg_200'].mean()
        train_avg200_std = train_data['avg_200'].std()   
        study_period['Normalized_Avg_200'] = study_period['avg_200'].apply(normalize_data,args=(train_avg200_mean,train_avg200_std))

        study_period.to_csv('study_period/'+str(file_prefix)+'_study_period_'+str(i)+'.csv')

generate_normalized_study_period(df_large_input, 'large')
generate_normalized_study_period(df_small_input, 'small')