import yfinance as yf
import pandas as pd

tickerStrings = ['EQNR.OL','DNB.OL','TEL.OL','MOWI.OL','YAR.OL','NHY.OL','ORK.OL','TOM.OL','STB.OL','GJF.OL','NEL.OL','ADE.OL','BAKKA.OL','SCATC.OL','SALM.OL','SCHB.OL','SCHA.OL','NOD.OL','AKRBP.OL','ENTRA.OL','LSG.OL','SUBC.OL','SRBNK.OL','BRG.OL','KOG.OL','TGS.OL','AKER.OL','VEI.OL','NOFI.OL','ATEA.OL','AFG.OL','ELK.OL','EPR.OL','PEXIP.OL','FKRFT.OL','HEX.OL','CRAYN.OL','RECSI.OL','FRO.OL','BOUV.OL','DNO.OL','BONHR.OL','BWLPG.OL','GOGL.OL','GSF.OL','PHO.OL','ABG.OL','KIT.OL','MEDI.OL','AZT.OL','AKSO.OL','KID.OL','MPCC.OL','WAWI.OL','SNI.OL','B2H.OL','KOA.OL','NAS.OL','XXL.OL','VOW.OL','AGAS.OL','BGBIO.OL','NANOV.OL','CARA.OL','IDEX.OL','ACR.OL','SATS.OL','PCIB.OL']
df_list = list()
for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period='max')
    data['ticker'] = ticker  # add this column becasue the dataframe doesn't contain a column with the ticker
    df_list.append(data)

# combine all dataframes into a single dataframe
df = pd.concat(df_list)

# save to csv
df.to_csv('largeticker.csv')
