import path
import pandas as pd
import datetime
import statistics
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from scipy.stats import norm
import numpy as np

stockCount = 5
neturalCount = 4
long_prob = (stockCount * -1)
short_prob = stockCount
netural_long_prob = neturalCount
netural_short_prob = neturalCount
plot_colors = ['red', 'blue', 'purple', 'green', 'cyan', 'orange', 'pink']
avg_10y_bonds_2013_2020 = 0.0172875

class Portfolio:
    def __init__(self, name, predictions) -> None:
        self.name = name
        self.predictions = predictions
        #long
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

        self.long_cum_rebalance = 0.0

        #short
        self.short_only_profit_nocost = {}
        self.short_only_accuprofit_nocost = {}
        self.short_only_nocost_accu_profit = 1
        self.short_only_nocost_return = 0
        self.short_only_nocost_sd = 0

        self.short_only_profit_cost = {}
        self.short_only_accuprofit_cost = {}
        self.short_only_cost_accu_profit = 1
        self.short_only_cost_return = 0
        self.short_only_cost_sd = 0

        self.short_cum_rebalance = 0.0

        #netural
        self.netural_only_profit_nocost = {}
        self.netural_only_accuprofit_nocost = {}
        self.netural_only_nocost_accu_profit = 1
        self.netural_only_nocost_return = 0
        self.netural_only_nocost_sd = 0

        self.netural_only_profit_cost = {}
        self.netural_only_accuprofit_cost = {}
        self.netural_only_cost_accu_profit = 1
        self.netural_only_cost_return = 0
        self.netural_only_cost_sd = 0

        self.netural_cum_rebalance = 0.0

class MDD:
    def __init__(self, peak_return, bottom_return, mdd) -> None:
        self.peak_return = peak_return
        self.bottom_return = bottom_return
        self.mdd = mdd

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

def calculate_returns(pf, returns, benchmark=False):
    date_list = list(pf.predictions.Date.unique())
    for date in date_list[:-1]:
        # No cost
        data_date = pf.predictions[pf.predictions.Date == date]
        long_part = data_date
        short_part = data_date
        if (benchmark is False):
            long_part = data_date.iloc[long_prob:]
            short_part = data_date.iloc[:short_prob]
        long_df = long_part.merge(returns,how='left',on = ['Date','Ticker'])
        short_df = short_part.merge(returns,how='left',on = ['Date', 'Ticker'])
        long_daily_profit = long_df['Returns'].mean()
        pf.long_only_nocost_accu_profit += long_daily_profit
        pf.long_only_profit_nocost.update({date:long_daily_profit})
        pf.long_only_accuprofit_nocost.update({date:pf.long_only_nocost_accu_profit})

        short_daily_profit = (-1) * (short_df['Returns'].mean())
        pf.short_only_nocost_accu_profit += short_daily_profit
        pf.short_only_profit_nocost.update({date:short_daily_profit})
        pf.short_only_accuprofit_nocost.update({date:pf.short_only_nocost_accu_profit})

        # Cost
        date_index = date_list.index(date)
        long_part_next = pf.predictions[pf.predictions.Date == date_list[date_index+1]]
        short_part_next = pf.predictions[pf.predictions.Date == date_list[date_index+1]]
        if (benchmark is False):
            long_part_next = long_part_next.iloc[long_prob:]
            short_part_next = short_part_next.iloc[:short_prob]
        long_ds_1 = long_part['Ticker']
        long_ds_2 = long_part_next['Ticker']
        long_rebalance_perc = rebalance_percentage(long_ds_1,long_ds_2)
        pf.long_cum_rebalance += long_rebalance_perc
        long_daily_profit_with_cost = (long_daily_profit - (0.00035*stockCount*2*long_rebalance_perc))
        pf.long_only_cost_accu_profit += long_daily_profit_with_cost
        pf.long_only_profit_cost.update({date:long_daily_profit_with_cost})
        pf.long_only_accuprofit_cost.update({date:pf.long_only_cost_accu_profit})

        short_ds_1 = short_part['Ticker']
        short_ds_2 = short_part_next['Ticker']
        short_rebalance_perc = rebalance_percentage(short_ds_1,short_ds_2)
        pf.short_cum_rebalance += long_rebalance_perc
        short_daily_profit_with_cost = (short_daily_profit - (0.00035*stockCount*2*short_rebalance_perc))
        pf.short_only_cost_accu_profit += short_daily_profit_with_cost
        pf.short_only_profit_cost.update({date:short_daily_profit_with_cost})
        pf.short_only_accuprofit_cost.update({date:pf.short_only_cost_accu_profit})
        
    #Long
    pf.long_only_nocost_return = (pf.long_only_nocost_accu_profit)/(len(date_list)-1)
    pf.long_only_nocost_sd = statistics.stdev(pf.long_only_profit_nocost.values())

    pf.long_only_cost_return = (pf.long_only_cost_accu_profit)/(len(date_list)-1)
    pf.long_only_cost_sd = statistics.stdev(pf.long_only_profit_cost.values())

    #Short
    pf.short_only_nocost_return = (pf.short_only_nocost_accu_profit)/(len(date_list)-1)
    pf.short_only_nocost_sd = statistics.stdev(pf.short_only_profit_nocost.values())

    pf.short_only_cost_return = (pf.short_only_cost_accu_profit)/(len(date_list)-1)
    pf.short_only_cost_sd = statistics.stdev(pf.short_only_profit_cost.values())

def calc_trades(cum_rebalance):
    return int(cum_rebalance * stockCount * 2) # buy and sell

def calc_avg_annual_returns(returns, trading_days, trading_year):
    fraction_year = trading_days / trading_year
    return returns / fraction_year

def calc_avg_annual_stdev(daily_stdev, trading_year):
    return daily_stdev * (math.sqrt(trading_year))

def calc_sharpe_ratio(avg_annual_returns, avg_annual_stdev):
    annual_returns_minus_free_risk = avg_annual_returns - avg_10y_bonds_2013_2020
    return annual_returns_minus_free_risk / avg_annual_stdev

def calc_value_at_risk(sig_interval, avg_annual_returns, avg_annual_stdev):
    return norm.ppf(sig_interval, avg_annual_returns, avg_annual_stdev)

def calc_max_drawdown(accuprofit):
    xs = list(accuprofit.values())
    bottom_point = np.argmax((np.maximum.accumulate(xs) - xs) / np.maximum.accumulate(xs))
    peak_point = np.argmax(xs[:bottom_point])
    peak_return = xs[peak_point]
    bottom_return = xs[bottom_point]
    mdd = (bottom_return - peak_return) / peak_return
    return MDD(peak_return, bottom_return, mdd)

def create_data_frame(name_list, start_date_list, end_date_list, trading_days_list, trades_list, accrued_list, avg_annual_return_list, avg_daily_list, 
                        avg_annual_stdev_list, stdev_list, sharpe_ratio_list, value_at_risk_list, mdd_list, mdd_peak_list, mdd_bottom_list):
    return pd.DataFrame(
    {'Name': name_list,
    'StartDate': start_date_list,
    'EndDate': end_date_list,
    'TradingDays': trading_days_list,
    'Trades': trades_list,
    'SumReturns': accrued_list,
    'AvgAnnualReturns': avg_annual_return_list,
    'AvgDailyReturns': avg_daily_list,
    'AvgAnnualStDev': avg_annual_stdev_list,
    'DailyStDev': stdev_list,
    'SharpeRatio': sharpe_ratio_list,
    'AnnualVaR95': value_at_risk_list,
    'Mdd': mdd_list,
    'MddPeakPoint': mdd_peak_list,
    'MddBottomPoint': mdd_bottom_list
    })

def portfolio_long_stats(portfolio_long_list):
    name_list = list()
    start_date_list = list()
    end_date_list = list()
    trading_days_list = list()
    trades_list = list()
    accrued_list = list()
    avg_daily_list = list()
    stdev_list = list()
    avg_annual_return_list = list()
    avg_annual_stdev_list = list()
    sharpe_ratio_list = list()
    value_at_risk_list = list()
    mdd_list = list()
    mdd_peak_list = list()
    mdd_bottom_list = list()
    for pf in portfolio_long_list:
        date_list = list(pf.predictions.Date.unique())
        trading_days = len(date_list) -1
        trading_year = 250.0
        name_list.append(pf.name + "_no_cost")
        name_list.append(pf.name + "_cost")
        start_date_list.append(date_list[0])
        start_date_list.append(date_list[0])
        end_date_list.append(date_list[-1])
        end_date_list.append(date_list[-1])
        trading_days_list.append(trading_days)
        trading_days_list.append(trading_days)
        trades = calc_trades(pf.long_cum_rebalance)
        trades_list.append(trades)
        trades_list.append(trades)
        accrued_list.append(pf.long_only_nocost_accu_profit)
        accrued_list.append(pf.long_only_cost_accu_profit)
        avg_daily_list.append(pf.long_only_nocost_return)
        avg_daily_list.append(pf.long_only_cost_return)
        stdev_list.append(pf.long_only_nocost_sd)
        stdev_list.append(pf.long_only_cost_sd)
        avg_annual_returns_no_cost = calc_avg_annual_returns(pf.long_only_nocost_accu_profit, trading_days, trading_year)
        avg_annual_returns_cost = calc_avg_annual_returns(pf.long_only_cost_accu_profit, trading_days, trading_year)
        avg_annual_return_list.append(avg_annual_returns_no_cost)
        avg_annual_return_list.append(avg_annual_returns_cost)
        avg_annual_stdev_no_cost = calc_avg_annual_stdev(pf.long_only_nocost_sd, trading_year)
        avg_annual_stdev_cost = calc_avg_annual_stdev(pf.long_only_cost_sd, trading_year)
        avg_annual_stdev_list.append(avg_annual_stdev_no_cost)
        avg_annual_stdev_list.append(avg_annual_stdev_cost)
        sharpe_ratio_no_cost = calc_sharpe_ratio(avg_annual_returns_no_cost, avg_annual_stdev_no_cost)
        sharpe_ratio_cost = calc_sharpe_ratio(avg_annual_returns_cost, avg_annual_stdev_cost)
        sharpe_ratio_list.append(sharpe_ratio_no_cost)
        sharpe_ratio_list.append(sharpe_ratio_cost)
        value_at_risk_list.append(calc_value_at_risk(0.05, avg_annual_returns_no_cost, avg_annual_stdev_no_cost))
        value_at_risk_list.append(calc_value_at_risk(0.05, avg_annual_returns_cost, avg_annual_stdev_cost))
        mdd_no_cost = calc_max_drawdown(pf.long_only_accuprofit_nocost)
        mdd_cost = calc_max_drawdown(pf.long_only_accuprofit_cost)
        mdd_list.append(mdd_no_cost.mdd)
        mdd_list.append(mdd_cost.mdd)
        mdd_peak_list.append(mdd_no_cost.peak_return)
        mdd_peak_list.append(mdd_cost.peak_return)
        mdd_bottom_list.append(mdd_no_cost.bottom_return)
        mdd_bottom_list.append(mdd_cost.bottom_return)
    return create_data_frame(name_list, start_date_list, end_date_list, trading_days_list, trades_list, accrued_list, avg_annual_return_list, avg_daily_list, 
                        avg_annual_stdev_list, stdev_list, sharpe_ratio_list, value_at_risk_list, mdd_list, mdd_peak_list, mdd_bottom_list)

def portfolio_short_stats(portfolio_short_list):
    name_list = list()
    start_date_list = list()
    end_date_list = list()
    trading_days_list = list()
    trades_list = list()
    accrued_list = list()
    avg_daily_list = list()
    stdev_list = list()
    avg_annual_return_list = list()
    avg_annual_stdev_list = list()
    sharpe_ratio_list = list()
    value_at_risk_list = list()
    mdd_list = list()
    mdd_peak_list = list()
    mdd_bottom_list = list()
    for pf in portfolio_short_list:
        date_list = list(pf.predictions.Date.unique())
        trading_days = len(date_list) -1
        trading_year = 250.0
        name_list.append(pf.name + "_no_cost")
        name_list.append(pf.name + "_cost")
        start_date_list.append(date_list[0])
        start_date_list.append(date_list[0])
        end_date_list.append(date_list[-1])
        end_date_list.append(date_list[-1])
        trading_days_list.append(trading_days)
        trading_days_list.append(trading_days)
        trades = calc_trades(pf.short_cum_rebalance)
        trades_list.append(trades)
        trades_list.append(trades)
        accrued_list.append(pf.short_only_nocost_accu_profit)
        accrued_list.append(pf.short_only_cost_accu_profit)
        avg_daily_list.append(pf.short_only_nocost_return)
        avg_daily_list.append(pf.short_only_cost_return)
        stdev_list.append(pf.short_only_nocost_sd)
        stdev_list.append(pf.short_only_cost_sd)
        avg_annual_returns_no_cost = calc_avg_annual_returns(pf.short_only_nocost_accu_profit, trading_days, trading_year)
        avg_annual_returns_cost = calc_avg_annual_returns(pf.short_only_cost_accu_profit, trading_days, trading_year)
        avg_annual_return_list.append(avg_annual_returns_no_cost)
        avg_annual_return_list.append(avg_annual_returns_cost)
        avg_annual_stdev_no_cost = calc_avg_annual_stdev(pf.short_only_nocost_sd, trading_year)
        avg_annual_stdev_cost = calc_avg_annual_stdev(pf.short_only_cost_sd, trading_year)
        avg_annual_stdev_list.append(avg_annual_stdev_no_cost)
        avg_annual_stdev_list.append(avg_annual_stdev_cost)
        sharpe_ratio_no_cost = calc_sharpe_ratio(avg_annual_returns_no_cost, avg_annual_stdev_no_cost)
        sharpe_ratio_cost = calc_sharpe_ratio(avg_annual_returns_cost, avg_annual_stdev_cost)
        sharpe_ratio_list.append(sharpe_ratio_no_cost)
        sharpe_ratio_list.append(sharpe_ratio_cost)
        value_at_risk_list.append(calc_value_at_risk(0.05, avg_annual_returns_no_cost, avg_annual_stdev_no_cost))
        value_at_risk_list.append(calc_value_at_risk(0.05, avg_annual_returns_cost, avg_annual_stdev_cost))
        mdd_no_cost = calc_max_drawdown(pf.short_only_accuprofit_nocost)
        mdd_cost = calc_max_drawdown(pf.short_only_accuprofit_cost)
        mdd_list.append(mdd_no_cost.mdd)
        mdd_list.append(mdd_cost.mdd)
        mdd_peak_list.append(mdd_no_cost.peak_return)
        mdd_peak_list.append(mdd_cost.peak_return)
        mdd_bottom_list.append(mdd_no_cost.bottom_return)
        mdd_bottom_list.append(mdd_cost.bottom_return)
    return create_data_frame(name_list, start_date_list, end_date_list, trading_days_list, trades_list, accrued_list, avg_annual_return_list, avg_daily_list, 
                        avg_annual_stdev_list, stdev_list, sharpe_ratio_list, value_at_risk_list, mdd_list, mdd_peak_list, mdd_bottom_list)

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
lstm1_8i_large_predictions = pd.read_csv('prediction/lstm1_8i_large_predictions.csv',index_col = 0)
gru1_large_predictions = pd.read_csv('prediction/gru1_large_predictions.csv',index_col = 0)
gru1_8i_large_predictions = pd.read_csv('prediction/gru1_8i_large_predictions.csv',index_col = 0)

lstm1_large_predictions = generate_probability(lstm1_large_predictions)
lstm2_large_predictions = generate_probability(lstm2_large_predictions)
lstm3_large_predictions = generate_probability(lstm3_large_predictions)
lstm1_8i_large_predictions = generate_probability(lstm1_8i_large_predictions)
gru1_large_predictions = generate_probability(gru1_large_predictions)
gru1_8i_large_predictions = generate_probability(gru1_8i_large_predictions)

benchmark_large = Portfolio('benchmark_large', lstm1_large_predictions)
pf_lstm1_large = Portfolio('lstm1_large', lstm1_large_predictions)
pf_lstm2_large = Portfolio('lstm2_large', lstm2_large_predictions)
pf_lstm3_large = Portfolio('lstm3_large', lstm3_large_predictions)
pf_lstm1_8i_large = Portfolio('lstm1_8i_large', lstm1_8i_large_predictions)
pf_gru1_large = Portfolio('gru1_large', gru1_large_predictions)
pf_gru1_8i_large= Portfolio('gru1_8i_large', gru1_8i_large_predictions)

# Small cap
lstm1_small_predictions = pd.read_csv('prediction/lstm1_small_predictions.csv',index_col = 0)
lstm2_small_predictions = pd.read_csv('prediction/lstm2_small_predictions.csv',index_col = 0)
lstm3_small_predictions = pd.read_csv('prediction/lstm3_small_predictions.csv',index_col = 0)
lstm1_8i_small_predictions = pd.read_csv('prediction/lstm1_8i_small_predictions.csv',index_col = 0)
gru1_small_predictions = pd.read_csv('prediction/gru1_small_predictions.csv',index_col = 0)
gru1_8i_small_predictions = pd.read_csv('prediction/gru1_8i_small_predictions.csv',index_col = 0)

lstm1_small_predictions = generate_probability(lstm1_small_predictions)
lstm2_small_predictions = generate_probability(lstm2_small_predictions)
lstm3_small_predictions = generate_probability(lstm3_small_predictions)
lstm1_8i_small_predictions = generate_probability(lstm1_8i_small_predictions)
gru1_small_predictions = generate_probability(gru1_small_predictions)
gru1_8i_small_predictions = generate_probability(gru1_8i_small_predictions)

benchmark_small = Portfolio('benchmark_small', lstm1_small_predictions)
pf_lstm1_small = Portfolio('lstm1_small', lstm1_small_predictions)
pf_lstm2_small = Portfolio('lstm2_small', lstm2_small_predictions)
pf_lstm3_small = Portfolio('lstm3_small', lstm3_small_predictions)
pf_lstm1_8i_small = Portfolio('lstm1_8i_small', lstm1_8i_small_predictions)
pf_gru1_small = Portfolio('gru1_small', gru1_small_predictions)
pf_gru1_8i_small = Portfolio('gru1_8i_small', gru1_8i_small_predictions)

# Entire portfolio list
#portfolio_list = [pf_lstm1_small]
portfolio_list = [benchmark_large, pf_lstm1_large, pf_lstm2_large, pf_lstm3_large, pf_lstm1_8i_large, pf_gru1_large, pf_gru1_8i_large,
                  benchmark_small, pf_lstm1_small, pf_lstm2_small, pf_lstm3_small, pf_lstm1_8i_small, pf_gru1_small, pf_gru1_8i_small]

for pf in portfolio_list:
    print("Calculating long only returns for portfolio \"" + pf.name + "\" from " +
              datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') + "...")
    if ("benchmark" in pf.name):
        print("This is interpreted as a benchmark portfolio. Calculate returns on all tickers")
        calculate_returns(pf, returns, True)
    else:
        calculate_returns(pf, returns)

print("Collecting long statistics...")
df_long_stat = portfolio_long_stats(portfolio_list)
df_long_stat = df_long_stat.sort_values(['SumReturns'], ascending = False)
print(df_long_stat)
df_long_stat.to_csv('pf_long_statistics.csv')

print("Collecting short statistics...")
df_short_stat = portfolio_short_stats(portfolio_list)
df_short_stat = df_short_stat.sort_values(['SumReturns'], ascending = False)
print(df_short_stat)
df_short_stat.to_csv('pf_short_statistics.csv')


portfolio_list_large = [pf_lstm1_large, pf_lstm2_large, pf_lstm3_large, pf_lstm1_8i_large, pf_gru1_large, pf_gru1_8i_large]
portfolio_list_small = [pf_lstm1_small, pf_lstm2_small, pf_lstm3_small, pf_lstm1_8i_small, pf_gru1_small, pf_gru1_8i_small]
portfolio_list_mix = [pf_lstm1_small, pf_lstm2_small, pf_lstm2_large, pf_gru1_8i_large]

print("Start drawing...")
# Plot line chart for cumulative return from 2012 - 2020 for all long large cap predictions, prior to transaction cost
for i in range(len(portfolio_list_large)):
    pf = portfolio_list_large[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_nocost.values(), color = color, label = pf.name)
plt.plot(list(benchmark_large.predictions.Date.unique())[:-1], benchmark_large.long_only_accuprofit_nocost.values(), color = 'brown', label = benchmark_large.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Cumulative Return On Large Cap No Cost')
plt.tight_layout()
plt.savefig('plots/cumulative_large_cap_return_no_cost.png')
plt.clf()

# Plot line chart for cumulative return from 2012 - 2020 for all long small cap predictions, prior to transaction cost
for i in range(len(portfolio_list_small)):
    pf = portfolio_list_small[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_nocost.values(), color = color, label = pf.name)
plt.plot(list(benchmark_small.predictions.Date.unique())[:-1], benchmark_small.long_only_accuprofit_nocost.values(), color = 'brown', label = benchmark_small.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Cumulative Return On Small Cap No Cost')
plt.tight_layout()
plt.savefig('plots/cumulative_small_cap_return_no_cost.png')
plt.clf()

# Plot line chart for cumulative return from 2012 - 2020 for all long large cap predictions, prior to transaction cost
for i in range(len(portfolio_list_large)):
    pf = portfolio_list_large[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_cost.values(), color = color, label = pf.name)
plt.plot(list(benchmark_large.predictions.Date.unique())[:-1], benchmark_large.long_only_accuprofit_nocost.values(), color = 'brown', label = benchmark_large.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Cumulative Return On Large Cap With Cost')
plt.tight_layout()
plt.savefig('plots/cumulative_large_cap_return_with_cost.png')
plt.clf()

# Plot line chart for cumulative return from 2012 - 2020 for all long small cap predictions, prior to transaction cost
for i in range(len(portfolio_list_small)):
    pf = portfolio_list_small[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_cost.values(), color = color, label = pf.name)
plt.plot(list(benchmark_small.predictions.Date.unique())[:-1], benchmark_small.long_only_accuprofit_nocost.values(), color = 'brown', label = benchmark_small.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Cumulative Return On Small Cap With Cost')
plt.tight_layout()
plt.savefig('plots/cumulative_small_cap_return_with_cost.png')
plt.clf()

# Mix
for i in range(len(portfolio_list_mix)):
    pf = portfolio_list_mix[i]
    color = plot_colors[i]
    date_list = list(pf.predictions.Date.unique())
    date_ar = date_list[:-1]
    plt.plot(date_ar, pf.long_only_accuprofit_nocost.values(), color = color, label = pf.name)
plt.plot(list(benchmark_small.predictions.Date.unique())[:-1], benchmark_small.long_only_accuprofit_nocost.values(), color = 'brown', label = benchmark_small.name)
plt.plot(list(benchmark_large.predictions.Date.unique())[:-1], benchmark_large.long_only_accuprofit_nocost.values(), color = 'gray', label = benchmark_large.name)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(240))
plt.xticks(rotation=90)
plt.legend()
plt.ylabel('Best Cumulative Return On Mix Cap No Cost')
plt.tight_layout()
plt.savefig('plots/cumulative_best_mix_cap_return_no_cost.png')
plt.clf()

print("Finished drawing!")