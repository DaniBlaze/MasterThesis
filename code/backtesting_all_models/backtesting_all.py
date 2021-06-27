import path
import pandas as pd
import datetime
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

stockCount = 5
long_prob = (stockCount * -1)
short_prob = stockCount
plot_colors = ['red', 'blue', 'yellow', 'green', 'cyan', 'magneta', 'black']

class LongPortfolio:
    def __init__(self, name, predictions) -> None:
        self.name = name
        self.predictions = predictions
        self.long_only_profit_nocost = {}
        self.long_only_accuprofit_nocost = {}
        self.long_only_nocost_accu_profit = 1
        self.long_only_nocost_return = 0
        self.long_only_nocost_sd = 0

        self.long_only_profit_cost = {}
        self.long_only_accuprofit_cost = {}
        self.long_only_cost_accu_profit = 1
        self.long_only_cost_return = 0
        self.long_only_cost_sd = 0

def count_accu(K,df):
    count = 0
    for i in range(K):
        if df.iloc[i,-1] == df.iloc[i,-2]:
            count+=1
    for i in range(K):
        if df.iloc[-(i+1),-1]== df.iloc[-(i+1),-2]:
            count+=1
    return count

def classification(y):
    if y >=0.5:
        return 1
    else:
        return 0

def generate_probability(predictions):
    predictions['y_pred'] = predictions['y_prob'].apply(classification)
    data = predictions.sort_values(['Date','y_prob'],ascending = True)
    data = data[['Date','Ticker','y_prob','y_pred','y_true']]
    return data.reset_index(drop = True)

def rebalance_percentage(ds_1,ds_2):
    ds = pd.concat([ds_1,ds_2],axis = 0,ignore_index = True).reset_index(drop = True)
    unique_length = len(ds.value_counts().keys().tolist())
    non_unique = ds.size- unique_length
    rebalance_percentage = (ds_1.size-non_unique)/(ds_1.size)
    return rebalance_percentage

def calculate_long_only_returns(pf, returns, benchmark=False):
    date_list = list(pf.predictions.Date.unique())
    for date in date_list[:-1]:
        # No cost
        data_date = pf.predictions[pf.predictions.Date == date]
        long_part = data_date
        if (benchmark is False):
            long_part = data_date.iloc[long_prob:]
        long_df = long_part.merge(returns,how ='left',on = ['Date','Ticker'])
        daily_profit = long_df['Returns'].mean()
        pf.long_only_nocost_accu_profit += daily_profit
        pf.long_only_profit_nocost.update({date:daily_profit})
        pf.long_only_accuprofit_nocost.update({date:pf.long_only_nocost_accu_profit})

        # Cost
        date_index = date_list.index(date)
        long_part_next = pf.predictions[pf.predictions.Date == date_list[date_index+1]]
        if (benchmark is False):
            long_part_next = long_part_next.iloc[long_prob:]
        ds_1 = long_part['Ticker']
        ds_2 = long_part_next['Ticker']
        #print("daily_no_cost: " + str(daily_profit))
        daily_profit_with_cost = (daily_profit - (0.0002*rebalance_percentage(ds_1,ds_2)))
        #print("daily_with_cost: " + str(daily_profit_with_cost))
        pf.long_only_cost_accu_profit += daily_profit_with_cost
        pf.long_only_profit_cost.update({date:daily_profit_with_cost})
        pf.long_only_accuprofit_cost.update({date:pf.long_only_cost_accu_profit})
    
    pf.long_only_nocost_return = (pf.long_only_nocost_accu_profit - 1)/(len(date_list)-1)
    pf.long_only_nocost_sd = statistics.stdev(pf.long_only_profit_nocost.values())

    pf.long_only_cost_return = (pf.long_only_cost_accu_profit - 1)/(len(date_list)-1)
    pf.long_only_cost_sd = statistics.stdev(pf.long_only_profit_cost.values())

def portfolio_long_stats(portfolio_long_list):
    name_list = list()
    start_date_list = list()
    end_date_list = list()
    accrued_percentage_list = list()
    avg_daily_list = list()
    stdev_list = list()
    for pf in portfolio_long_list:
        date_list = list(pf.predictions.Date.unique())
        name_list.append(pf.name + "_no_cost")
        name_list.append(pf.name + "_cost")
        start_date_list.append(date_list[0])
        start_date_list.append(date_list[0])
        end_date_list.append(date_list[-1])
        end_date_list.append(date_list[-1])
        accrued_percentage_list.append((pf.long_only_nocost_accu_profit*100))
        accrued_percentage_list.append((pf.long_only_cost_accu_profit*100))
        avg_daily_list.append((pf.long_only_nocost_return * 100))
        avg_daily_list.append((pf.long_only_cost_return * 100))
        stdev_list.append((pf.long_only_nocost_sd*100))
        stdev_list.append((pf.long_only_cost_sd*100))
    return pd.DataFrame(
    {'Name': name_list,
    'StartDate': start_date_list,
    'EndDate': end_date_list,
    'ProfitPercentage': accrued_percentage_list,
    'AvgDailyPercentage': avg_daily_list,
    'StDev': stdev_list
    })

# Reading csv-file using a relative path, based on the folder structure of the github project
file_path = path.Path(__file__).parent / "return_data/returns.csv"
with file_path.open() as dataset_file:
    returns = pd.read_csv(dataset_file)
    returns.dropna(inplace=True)
    returns.reset_index(inplace=True, drop=True)
    returns = returns[['Date','Ticker','Returns']]

# Large cap
lstm1_large_predictions = pd.read_csv('prediction/lstm1_large_predictions.csv',index_col = 0)
lstm2_large_predictions = pd.read_csv('prediction/lstm2_large_predictions.csv',index_col = 0)
lstm3_large_predictions = pd.read_csv('prediction/lstm3_large_predictions.csv',index_col = 0)

lstm1_large_predictions = generate_probability(lstm1_large_predictions)
lstm2_large_predictions = generate_probability(lstm2_large_predictions)
lstm3_large_predictions = generate_probability(lstm3_large_predictions)

long_pf_lstm1_large = LongPortfolio('lstm1_large', lstm1_large_predictions)
long_pf_lstm1_large_benchmark = LongPortfolio('lstm1_large_benchmark', lstm1_large_predictions)
long_pf_lstm2_large = LongPortfolio('lstm2_large', lstm2_large_predictions)
long_pf_lstm3_large = LongPortfolio('lstm3_large', lstm3_large_predictions)

# Small cap
lstm1_small_predictions = pd.read_csv('prediction/lstm1_small_predictions.csv',index_col = 0)
lstm2_small_predictions = pd.read_csv('prediction/lstm2_small_predictions.csv',index_col = 0)
lstm3_small_predictions = pd.read_csv('prediction/lstm3_small_predictions.csv',index_col = 0)
lstm1_8i_small_predictions = pd.read_csv('prediction/lstm1_8i_small_predictions.csv',index_col = 0)


lstm1_small_predictions = generate_probability(lstm1_small_predictions)
lstm2_small_predictions = generate_probability(lstm2_small_predictions)
lstm3_small_predictions = generate_probability(lstm3_small_predictions)
lstm1_8i_small_predictions = generate_probability(lstm1_8i_small_predictions)

long_pf_lstm1_small = LongPortfolio('lstm1_small', lstm1_small_predictions)
long_pf_lstm1_small_benchmark = LongPortfolio('lstm1_small_benchmark', lstm1_small_predictions)
long_pf_lstm2_small = LongPortfolio('lstm2_small', lstm2_small_predictions)
long_pf_lstm3_small = LongPortfolio('lstm3_small', lstm3_small_predictions)
long_pf_lstm1_8i_small = LongPortfolio('lstm1_8i_small', lstm1_8i_small_predictions)

# Entire portfolio list
portfolio_list = [long_pf_lstm1_large]
#portfolio_list = [long_pf_lstm1_large, long_pf_lstm1_large_benchmark, long_pf_lstm2_large, long_pf_lstm3_large,
#                  long_pf_lstm1_small, long_pf_lstm1_small_benchmark, long_pf_lstm2_small, long_pf_lstm3_small, long_pf_lstm1_8i_small]

for pf in portfolio_list:
    print("Calculating long only returns for portfolio \"" + pf.name + "\" from " +
              datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + "...")
    if ("benchmark" in pf.name):
        print("This is interpreted as a benchmark portfolio. Calculate returns on all tickers")
        calculate_long_only_returns(pf, returns, True)
    else:
        calculate_long_only_returns(pf, returns)

print("Collecting profit statistics...")
df_long_stat = portfolio_long_stats(portfolio_list)
df_profit_percentage = df_long_stat.sort_values(['ProfitPercentage'], ascending = False)
print(df_profit_percentage)
df_profit_percentage.to_csv('pf_profit_percentage.csv')

'''
portfolio_list_large = [long_pf_lstm1_large, long_pf_lstm1_large_benchmark, long_pf_lstm2_large, long_pf_lstm3_large]
portfolio_list_small = [long_pf_lstm1_small, long_pf_lstm1_small_benchmark, long_pf_lstm2_small, long_pf_lstm3_small, long_pf_lstm1_8i_small]

print("Start drawing...")
# Plot line chart for accumulative return from 2012 - 2020 for all long large cap predictions, prior to transaction cost
for i in range(len(portfolio_list_large)):
    pf = portfolio_list_large[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_nocost.values(), color = color, label = pf.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Accumulative Return On Large Cap No Cost')
plt.tight_layout()
plt.savefig('plots/accumulative_large_cap_return_no_cost.png')
plt.clf()

# Plot line chart for accumulative return from 2012 - 2020 for all long small cap predictions, prior to transaction cost
for i in range(len(portfolio_list_small)):
    pf = portfolio_list_small[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_nocost.values(), color = color, label = pf.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Accumulative Return On Small Cap No Cost')
plt.tight_layout()
plt.savefig('plots/accumulative_small_cap_return_no_cost.png')
plt.clf()

# Plot line chart for accumulative return from 2012 - 2020 for all long large cap predictions, prior to transaction cost
for i in range(len(portfolio_list_large)):
    pf = portfolio_list_large[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_cost.values(), color = color, label = pf.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Accumulative Return On Large Cap With Cost')
plt.tight_layout()
plt.savefig('plots/accumulative_large_cap_return_with_cost.png')
plt.clf()

# Plot line chart for accumulative return from 2012 - 2020 for all long small cap predictions, prior to transaction cost
for i in range(len(portfolio_list_small)):
    pf = portfolio_list_small[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_cost.values(), color = color, label = pf.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Accumulative Return On Small Cap With Cost')
plt.tight_layout()
plt.savefig('plots/accumulative_small_cap_return_with_cost.png')
plt.clf()

print("Finished drawing!")
'''