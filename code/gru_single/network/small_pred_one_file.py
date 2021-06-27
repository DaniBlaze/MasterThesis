import pandas as pd

df = pd.concat(map(pd.read_csv, ['prediction/small_prediction_period_0.csv',
                                 'prediction/small_prediction_period_1.csv',
                                 'prediction/small_prediction_period_2.csv',
                                 'prediction/small_prediction_period_3.csv',
                                 'prediction/small_prediction_period_4.csv',
                                 'prediction/small_prediction_period_5.csv',
                                 'prediction/small_prediction_period_6.csv',
                                 'prediction/small_prediction_period_7.csv']))
df = df.drop(df.columns[0], axis=1)
print("saving...")
df.to_csv('prediction/small_cap_predictions.csv')
