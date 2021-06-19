import numpy
import yfinance as yf
import pandas as pd
import path
import math
pd.options.mode.chained_assignment = None  # default='warn'

all_tickers = ['2020.OL',
               'ABG.OL',
               'ADE.OL',
               'AFG.OL',
               'AKAST.OL',
               'AKER.OL',
               'AKBM.OL',
               'AKRBP.OL',
               'AKH.OL',
               'AKSO.OL',
               'AKVA.OL',
               'AMSC.OL',
               'AQUA.OL',
               'ARCH.OL',
               'AZT.OL',
               'ARCUS.OL',
               'AFK.OL',
               'ARR.OL',
               'ASTK.OL',
               'ATEA.OL',
               'ASA.OL',
               'AURG.OL',
               'AUSS.OL',
               'AGAS.OL',
               'AWDR.OL',
               'ACR.OL',
               'B2H.OL',
               'BAKKA.OL',
               'BELCO.OL',
               'BGBIO.OL',
               'BEWI.OL',
               'BONHR.OL',
               'BOR.OL',
               'BORR.OL',
               'BRG.OL',
               'BOUV.OL',
               'BWE.OL',
               'BWLPG.OL',
               'BWO.OL',
               'BMA.OL',
               'CADLR.OL',
               'CARA.OL',
               'CONTX.OL',
               'CRAYN.OL',
               'DLTX.OL',
               'DNB.OL',
               'DNO.OL',
               'DOF.OL',
               'EIOF.OL',
               'EMGS.OL',
               'ELK.OL',
               'ENDUR.OL',
               'ENSU.OL',
               'ENTRA.OL',
               'EQNR.OL',
               'EPR.OL',
               'FJORD.OL',
               'FKRFT.OL',
               'FLNG.OL',
               'FRO.OL',
               'FROY.OL',
               'GIG.OL',
               'RISH.OL',
               'GJF.OL',
               'GOGL.OL',
               'GOD.OL',
               'GSF.OL',
               'GYL.OL',
               'HAFNI.OL',
               'HAVI.OL',
               'HYARD.OL',
               'HEX.OL',
               'HBC.OL',
               'HSPG.OL',
               'IDEX.OL',
               'INFRO.OL',
               'INSR.OL',
               'IOX.OL',
               'ITERA.OL',
               'JIN.OL',
               'JAREN.OL',
               'KAHOT.OL',
               'KID.OL',
               'KIT.OL',
               'KMCP.OL',
               'KOMP.OL',
               'KOA.OL',
               'KOG.OL',
               'LSG.OL',
               'LINK.OL',
               'MGN.OL',
               'MSEIS.OL',
               'MEDI.OL',
               'MELG.OL',
               'MOWI.OL',
               'MPCC.OL',
               'MULTI.OL',
               'NAPA.OL',
               'NAVA.OL',
               'NKR.OL',
               'NEL.OL',
               'NEXT.OL',
               'NORBT.OL',
               'NANOV.OL',
               'NOD.OL',
               'NHY.OL',
               'NSKOG.OL',
               'NODL.OL',
               'NOL.OL',
               'NRS.OL',
               'NAS.OL',
               'NOR.OL',
               'NOFI.OL',
               'NPRO.OL',
               'NRC.OL',
               'NTS.OL',
               'OCY.OL',
               'OTS.OL',
               'ODL.OL',
               'ODF.OL',
               'ODFB.OL',
               'OKEA.OL',
               'OET.OL',
               'OLT.OL',
               'ORK.OL',
               'OTEC.OL',
               'PEN.OL',
               'PARB.OL',
               'PCIB.OL',
               'PSE.OL',
               'PEXIP.OL',
               'PGS.OL',
               'PHO.OL',
               'PLCS.OL',
               'POL.OL',
               'PLT.OL',
               'PRS.OL',
               'PROT.OL',
               'QFR.OL',
               'QEC.OL',
               'RAKP.OL',
               'REACH.OL',
               'RECSI.OL',
               'SAGA.OL',
               'SALM.OL',
               'SACAM.OL',
               'SADG.OL',
               'SASNO.OL',
               'SATS.OL',
               'SBANK.OL',
               'SCANA.OL',
               'SCATC.OL',
               'SCHA.OL',
               'SCHB.OL',
               'SDSD.OL',
               'SBX.OL',
               'SDRL.OL',
               'SSG.OL',
               'SBO.OL',
               'SHLF.OL',
               'SIOFF.OL',
               'SKUE.OL',
               'SOGN.OL',
               'SOLON.OL',
               'SOFF.OL',
               'MING.OL',
               'SRBNK.OL',
               'SOON.OL',
               'MORG.OL',
               'SOR.OL',
               'SVEG.OL',
               'SPOG.OL',
               'SNOR.OL',
               'SPOL.OL',
               'HELG.OL',
               'NONG.OL',
               'RING.OL',
               'SOAG.OL',
               'SNI.OL',
               'STB.OL',
               'STRO.OL',
               'SUBC.OL',
               'TRVX.OL',
               'TECH.OL',
               'TEL.OL',
               'TGS.OL',
               'TIETO.OL',
               'TOM.OL',
               'TOTG.OL',
               'TRE.OL',
               'ULTI.OL',
               'VEI.OL',
               'VISTN.OL',
               'VOLUE.OL',
               'VVL.OL',
               'VOW.OL',
               'WAWI.OL',
               'WSTEP.OL',
               'WWI.OL',
               'WWIB.OL',
               'WILS.OL',
               'XXL.OL',
               'YAR.OL',
               'ZAL.OL']

#all_tickers = ['EQNR.OL', 'NHY.OL']
input_tickers = ['^VIX', 'BZ=F', '^TNX', 'NOK=X'] # + Volume, + 50/200 moving avg
#small_cap_tickers = ['FKRFT.OL', 'PROT.OL']

def calculate_returns(ticker_data):
    returns_list = list()
    previous_ticker_day = None
    for ticker_day in ticker_data.itertuples():
        if previous_ticker_day == None:
            returns_list.append(
                (ticker_day.Close - ticker_day.Open)/ticker_day.Open)
        else:
            # 'Adj Close' column will be named _5 by namedTuples
            returns_list.append(
                (ticker_day._5 - previous_ticker_day._5)/previous_ticker_day._5)
        previous_ticker_day = ticker_day
    return returns_list

def add_moving_price_avg(ticker_data, days):
    counter = 0
    moving_avg_price_list = list()
    unique_dates = list(ticker_data.index.unique())
    for ticker_day in ticker_data.itertuples():
        if(counter >= days):
            start = counter-days
            sub_dates = unique_dates[start : counter]
            sub_ticker_data = ticker_data[ticker_data.index.isin(sub_dates)]
            avg_price = sub_ticker_data['Adj Close'].mean()
            moving_avg_price_list.append(avg_price)
            counter +=1
        else:
            moving_avg_price_list.append(numpy.nan)
            counter +=1
    return moving_avg_price_list

def add_year_column(df_ticker_data):
    data_frame_output = pd.DataFrame()
    dates = list(df_ticker_data.index.unique())
    dates.sort()
    for date in dates:
        sub_date_data = df_ticker_data[df_ticker_data.index == date]
        sub_date_data['Year'] = date.year
        data_frame_output = pd.concat([data_frame_output, sub_date_data], ignore_index=False)
    return data_frame_output

def scrape_ticker_data(cap_tickers):
    df_ticker_data = list()
    for ticker in cap_tickers:
        ticker_data = yf.download(ticker, group_by="Ticker", period='max')
        # add ticker column because the dataframe doesn't contain a column with the ticker
        ticker_data['Ticker'] = ticker
        # calculate and add returns column
        ticker_data['Returns'] = calculate_returns(ticker_data)
        # add moving avg
        ticker_data['avg_50'] = add_moving_price_avg(ticker_data, 50)
        ticker_data['avg_200'] = add_moving_price_avg(ticker_data, 200)
        df_ticker_data.append(ticker_data)
    df_concat = pd.concat(df_ticker_data)
    print("Adding Year column")
    return add_year_column(df_concat)

def scrape_extra_input_data(input_tickers):
    df_ticker_data = list()
    for ticker in input_tickers:
        ticker_data = yf.download(ticker, group_by="Ticker", period='max')
        # add ticker column because the dataframe doesn't contain a column with the ticker
        ticker_data['Ticker'] = ticker
        df_ticker_data.append(ticker_data)
    df_concat = pd.concat(df_ticker_data)
    return df_concat

def generate_finalized_input_data(input_data):
    input_finalized = pd.DataFrame()
    tickers = list(input_data.Ticker.unique())
    for ticker in tickers:
        ticker_data = input_data[input_data.Ticker == ticker]
        ticker_name = str(ticker_data.iloc[0,-1])
        print(ticker_name)
        ticker_data[ticker_name] = ticker_data['Adj Close']
        ticker_data = ticker_data[[ticker_name]]
        if(input_finalized.empty):
            input_finalized = ticker_data
        else:
            input_finalized = pd.concat([input_finalized, ticker_data], axis=1)
    return input_finalized


def merge_market_cap(df_output):
    # Reading csv-file using a relative path, based on the folder structure of the github project
    file_path = path.Path(__file__).parent / "../static/marketCapAllShares.csv"
    with file_path.open() as dataset_file:
        df_static_market_cap_per_year = pd.read_csv(dataset_file, delimiter=";")
    return pd.merge(df_output, df_static_market_cap_per_year, how='left', on=['Ticker', 'Year']).set_index(df_output.index)

print("Scraping historic data for all tickers with size " + str(len(all_tickers)))
df_scraped_data = scrape_ticker_data(all_tickers)
print("Finished scraping data for all cap tickers")

print("Scraping input data with size " + str(len(input_tickers)))
df_scraped_input = scrape_extra_input_data(input_tickers)
print("Concatinating adjusted close for all input data")
df_finalized_input = generate_finalized_input_data(df_scraped_input)

print("Merging market cap into existing dataframe")
df_merged_with_market_cap = merge_market_cap(df_scraped_data)

print("Merging existing dataframe with input data")
df_finalized = pd.merge(df_merged_with_market_cap, df_finalized_input, how='left', left_index=True, right_index=True).set_index(df_merged_with_market_cap.index)

# save to csv
df_finalized.to_csv('scrapedData.csv')