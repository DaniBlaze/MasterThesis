import path
import pandas as pd

file_prefix = 'small'
years = 9
for m in range(years):
    file_path = path.Path(__file__).parent / 'study_period/'+str(file_prefix)+'_study_period_'+str(m)+'.csv'
    with file_path.open() as dataset_file:
        data_input = pd.read_csv(dataset_file)
    data_input = data_input[['Date', 'Ticker', 'Normalized_Returns', 'Target']]
    data_input = data_input.sort_values(['Date','Ticker'], ascending = True)
    unique_dates = list(data_input.Date.unique())
    tickers = list(data_input.Ticker.unique())
    tickers.sort()

    final_training_set = pd.DataFrame()
    for ticker in tickers:
        train_output = pd.DataFrame()
        for i in range(510):
            sub_dates = unique_dates[i : (i+241)]
            ticker_sub_data = data_input[data_input.Ticker == ticker]
            ticker_date_data = ticker_sub_data[ticker_sub_data.Date.isin(sub_dates)]
            ticker_date_data = ticker_date_data.reset_index(drop = True)
            if len(ticker_date_data.index) >= 241:
                ticker_date_data = ticker_date_data.transpose()
                ticker_name = ticker_date_data.iloc[1,0]
                date = ticker_date_data.iloc[0,-1]
                target = ticker_date_data.iloc[-1,-1]

                ticker_date_data = ticker_date_data.drop(['Date','Ticker','Target'],axis = 0)
                ticker_date_data = ticker_date_data.iloc[:,0:-1]
                ticker_date_data = ticker_date_data.stack().to_frame().T
                ticker_date_data.columns = range(240)

                ticker_date_data['target'] = target
                ticker_date_data['ticker'] = ticker_name
                ticker_date_data['date'] = date
                ticker_date_data = ticker_date_data.reset_index(drop= True)
                train_output = pd.concat([train_output,ticker_date_data],ignore_index = True)
        final_training_set = pd.concat([final_training_set,train_output],ignore_index = True)
    print('Saving training set for year ' + str(m) + ' of ' + str(years-1))
    final_training_set.to_csv('finalized_dataset/'+str(file_prefix)+'_training_set_'+str(m)+'.csv')

    final_test_set = pd.DataFrame()
    for ticker in tickers:
        test_output = pd.DataFrame()
        for i in range(510,760):
            sub_dates = unique_dates[i : (i+241)]
            ticker_sub_data = data_input[data_input.Ticker == ticker]
            ticker_date_data = ticker_sub_data[ticker_sub_data.Date.isin(sub_dates)]
            ticker_date_data = ticker_date_data.reset_index(drop = True)
            if len(ticker_date_data.index) >= 241:
                ticker_date_data = ticker_date_data.transpose()
                ticker_name = ticker_date_data.iloc[1,0]
                date = ticker_date_data.iloc[0,-1]
                target = ticker_date_data.iloc[-1,-1]

                ticker_date_data = ticker_date_data.drop(['Date','Ticker','Target'],axis = 0)
                ticker_date_data = ticker_date_data.iloc[:,0:-1]
                ticker_date_data = ticker_date_data.stack().to_frame().T
                ticker_date_data.columns = range(240)

                ticker_date_data['target'] = target
                ticker_date_data['ticker'] = ticker_name
                ticker_date_data['date'] = date
                ticker_date_data = ticker_date_data.reset_index(drop= True)
                test_output = pd.concat([test_output,ticker_date_data],ignore_index = True)
        final_test_set = pd.concat([final_test_set,test_output],ignore_index = True)
    print('Saving test set for year ' + str(m) + ' of ' + str(years-1))
    final_test_set.to_csv('finalized_dataset/'+str(file_prefix)+'_test_set_'+str(m)+'.csv')
print('Finished generating training and test datasets for '+str(file_prefix)+' cap')