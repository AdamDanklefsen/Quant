def getAlphaVantageOptionsData(startDate = "2020-01-01", nTickers = None):
    import pandas as pd
    import time
    import os

    # AVKEY = "EVECVIOQ5AGUUTK9"
    AVKEY = os.getenv('AlphaVantage_API_KEY')

    tickers = pd.read_json("./tickers.json").T

    BanList = ['CON', 'BRK-B']
    TickerList = [
        # Mega-cap tech (best options liquidity)
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
        
        # ETFs (most liquid)
        'SPY', 'QQQ', 'IWM',  # S&P 500, Nasdaq, Russell 2000
        
        # High IV / Volatility plays
        'AMD', 'NFLX', 'PLTR', 'COIN',
        
        # Financial
        'JPM', 'GS', 'BAC',
        
        # Consumer / Retail
        'WMT', 'COST', 'DIS',
        
        # Healthcare / Pharma
        'JNJ', 'UNH',
        
        # Energy / Industrials
        'XOM', 'BA'
    ]
    tickers = tickers[tickers['ticker'].isin(TickerList)]
    print(tickers)

    if nTickers is None:
        nTickers = len(tickers)

    totalData = pd.DataFrame()
    QuereiesUsed = 0
    for ticker in tickers['ticker'].values[:nTickers]:
        print(f"Fetching data for {ticker}...")

        tData, qT = FetchOptionsData(startDate, ticker, AVKEY)
        if tData.empty:
            print(f"No data fetched for {ticker}, skipping...")
            continue
        totalData = pd.concat([totalData, tData])
        QuereiesUsed += qT
        if QuereiesUsed != 0:
            print(f"Completed fetching data for {ticker}. Total Queries Used: {QuereiesUsed}")
        if QuereiesUsed >= 72:
            print("Reached Alpha Vantage query limit for the minute, Sleeping for 60 seconds...")
            time.sleep(60)
            QuereiesUsed = 0

    print(f"Total Queries Used: {QuereiesUsed}")

    # cast all number like columns to numeric
    for col in totalData.columns:
        print(f"Column {col} type: {totalData[col].dtype}")
    return totalData


def FetchOptionsData(startDate, ticker, apiKey):
    # Check if cached file exists
    import os
    import pandas as pd
    import time
    import shutil
    
    # Create ticker folder if it doesn't exist
    ticker_folder = f"D:/AlphaVantageData/Options/{ticker}"
    if not os.path.exists(ticker_folder):
        os.makedirs(ticker_folder)
    
    # Create finished folder if it doesn't exist
    finished_folder = f"D:/AlphaVantageData/OptionsFinished"
    if not os.path.exists(finished_folder):
        os.makedirs(finished_folder)
    
    if os.path.exists(f"{finished_folder}/{ticker}.csv"):
        df = pd.read_csv(f"{finished_folder}/{ticker}.csv",
                         index_col=['contractID', 'date'], parse_dates=['date', 'expiration'])
        df = df.infer_objects()
        df = df.reset_index().set_index(['contractID', 'date'])
        df.index.names = ['contractID', 'date']

        # Delete cache files to save space
        shutil.rmtree(ticker_folder)
        return df, 0
    else:
        import requests
        fullOptions = pd.DataFrame()
        QueryCount = 0
        start_time = time.time()
        print(f"Fetching Alpha Vantage data for {ticker}...")
        for day in pd.date_range(start=startDate, end=pd.Timestamp.today()):
            if day.day_name() in ['Saturday', 'Sunday']:
                # print(f"{day.strftime('%Y-%m-%d')} is a weekend, skipping...")
                continue
            
            # Check if day file exists
            day_str = day.strftime('%Y-%m-%d')
            day_file = f"{ticker_folder}/{ticker}_{day_str}.csv"
            if os.path.exists(day_file):
                options = pd.read_csv(day_file, index_col=['contractID', 'date'], parse_dates=['date', 'expiration'])
                options = options.infer_objects()
                options = options.reset_index().set_index(['contractID', 'date'])
                fullOptions = pd.concat([fullOptions, options])
                continue
            
            # print(f"Fetching options data for {ticker} on {day.strftime('%Y-%m-%d')}...")
            try:
                response = requests.get(
                    f"https://www.alphavantage.co/query?function=HISTORICAL_OPTIONS&symbol={ticker}&date={day_str}&apikey={apiKey}")
                QueryCount += 1
                if 'data' not in response.json():
                    print(f"No options data found for {ticker} on {day_str}: {response.json()}, skipping...")
                    continue
                options = pd.DataFrame(response.json()['data'])
            except Exception as e:
                print(f"Error fetching options data for {ticker} on {day_str}: {e}, skipping...")
                continue
            if len(options) == 0:
                print(f"No options data found for {ticker} on {day_str} ({day.day_name()}), skipping...")
                continue
            options.set_index(['contractID', 'date'], inplace=True)
            options = options.infer_objects()
            
            # Save day file
            options.to_csv(day_file)
            fullOptions = pd.concat([fullOptions, options])

            if QueryCount >= 72:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print(f"Elapsed time for {QueryCount} queries: {elapsed_time:.2f} seconds")
                print("Reached Alpha Vantage query limit for the minute, Sleeping for 60 seconds...")
                # progressSleep(60)
                QueryCount = 0

        fullOptions.to_csv(f"{finished_folder}/{ticker}.csv")
        return fullOptions, 1






def FetchTickerData(ticker, apiKey):
    # Check if cached finished file exists
    import os
    import pandas as pd
    
    if os.path.exists(f"../AlphaVantageData/Finished/{ticker}.csv"):
        df = pd.read_csv(f"../AlphaVantageData/Finished/{ticker}.csv",
                            index_col=['Ticker', 'Reported Date'],
                            parse_dates=['Reported Date', 'fiscalDateEnding'])
        df = df.infer_objects()
        df.index.names = ['Ticker', 'Reported Date']
        return df, 0
    else:
        print(f"Fetching Alpha Vantage data for {ticker}...")
        income, qI = FetchIncomeStatement(ticker, apiKey)
        if income.empty:
            print(f"No income statement data for {ticker}, skipping...")
            return pd.DataFrame(), qI
        income.index = pd.to_datetime(income.index)

        balance, qB = FetchBalanceSheet(ticker, apiKey)
        balance.index = pd.to_datetime(balance.index)

        earnings, qE = FetchEarnings(ticker, apiKey)
        if earnings.empty:
            print(f"No earnings data for {ticker}, skipping...")
            return pd.DataFrame(), qI + qB + qE
        earnings.index = pd.to_datetime(earnings.index)

        tData = pd.DataFrame(pd.concat([
            income[['totalRevenue', 'ebitda', 'netIncome']],
            balance[['totalAssets', 'totalLiabilities', 'totalShareholderEquity']],
            earnings[['reportedDate', 'reportedEPS', 'estimatedEPS', 'reportTime']]
        ], axis=1))
        tData['fiscalDateEnding'] = tData.index
        tData['Ticker'] = ticker

        tData = tData.set_index(['Ticker', 'reportedDate'])
        tData.index.names = ['Ticker', 'Reported Date']
        tData = tData.dropna(thresh=len(tData.columns) - 2)
        tData.to_csv(f"../AlphaVantageData/Finished/{ticker}.csv", index=True)
        return tData, (qI + qB + qE)


def FetchBalanceSheet(ticker, apiKey):
    # Check if cached file exists
    import os
    import pandas as pd

    if os.path.exists(f"../AlphaVantageData/BalanceSheets/{ticker}.csv"):
        df = pd.read_csv(f"../AlphaVantageData/BalanceSheets/{ticker}.csv", index_col='fiscalDateEnding', parse_dates=True)
        df = df.infer_objects()
        df.index.names = ['fiscalDateEnding']
        return df, 0
    else:
        import requests

        response = requests.get(
            f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={apiKey}")
        balance = pd.DataFrame(response.json()['quarterlyReports'])
        balance.set_index('fiscalDateEnding', inplace=True)
        balance = balance.infer_objects()
        balance.to_csv(f"../AlphaVantageData/BalanceSheets/{ticker}.csv")
        return balance, 1

def FetchEarnings(ticker, apiKey):
    # Check if cached file exists
    import os
    import pandas as pd

    if os.path.exists(f"../AlphaVantageData/Earnings/{ticker}.csv"):
        df = pd.read_csv(f"../AlphaVantageData/Earnings/{ticker}.csv", index_col='fiscalDateEnding', parse_dates=True)
        df = df.infer_objects()
        df.index.names = ['fiscalDateEnding']
        return df, 0
    else:
        import requests

        response = requests.get(
            f"https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={apiKey}")
        earnings = pd.DataFrame(response.json()['quarterlyEarnings'])
        if len(earnings) == 0:
            print(f"No Quarterly Earnings found for {ticker}")
            return pd.DataFrame(), 0
        earnings.set_index('fiscalDateEnding', inplace=True)
        earnings['estimatedEPS'] = pd.to_numeric(earnings['estimatedEPS'], errors='coerce')
        earnings = earnings.infer_objects()
        earnings.to_csv(f"../AlphaVantageData/Earnings/{ticker}.csv")
        return earnings, 1

def FetchIncomeStatement(ticker, apiKey):
    # Check if cached file exists
    import os
    import pandas as pd

    if os.path.exists(f"../AlphaVantageData/IncomeStatements/{ticker}.csv"):
        df = pd.read_csv(f"../AlphaVantageData/IncomeStatements/{ticker}.csv", index_col='fiscalDateEnding', parse_dates=True)
        df = df.infer_objects()
        df.index.names = ['fiscalDateEnding']
        return df, 0
    else:
        import requests

        response = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={apiKey}")
        if response.status_code != 200:
            raise Exception(f"Error fetching income statement for {ticker}: {response.status_code}")
        if 'quarterlyReports' not in response.json():
            print(f"No Quarterly Reports found for {ticker}")
            return pd.DataFrame(), 0
        if len(response.json()['quarterlyReports']) == 0:
            print(f"No Quarterly Reports found for {ticker}")
            return pd.DataFrame(), 0
        income = pd.DataFrame(response.json()['quarterlyReports'])
        income.set_index('fiscalDateEnding', inplace=True)
        income = income.infer_objects()
        income.to_csv(f"../AlphaVantageData/IncomeStatements/{ticker}.csv")
        return income, 1

def FetchShareData(ticker, apiKey):
    # Check if cached file exists
    import os
    import pandas as pd
    import numpy as np

    if os.path.exists(f"../AlphaVantageData/Shares/{ticker}.csv"):
        df = pd.read_csv(f"../AlphaVantageData/Shares/{ticker}.csv", index_col=['ticker','date'], parse_dates=['date'])
        df = df.infer_objects()
        df.index.names = ['ticker', 'date']
        return df, 0
    else:
        import requests

        response = requests.get(
            f"https://www.alphavantage.co/query?function=SHARES_OUTSTANDING&symbol={ticker}&apikey={apiKey}&outputsize=full")
        if response.status_code != 200:
            raise Exception(f"Error fetching share data for {ticker}: {response.status_code}")
        if 'data' not in response.json():
            print(f"No Time Series data found for {ticker}")
            return pd.DataFrame(), 0
        if len(response.json()['data']) == 0:
            print(f"No Time Series data found for {ticker}")
            return pd.DataFrame(), 0
        if 'shares_outstanding_diluted' not in response.json()['data'][0]:
            print(f"No Shares Outstanding Diluted data found for {ticker}")
            return pd.DataFrame(), 0
        shares = pd.DataFrame(
            {
                'diluted_shares': {
                    pd.to_datetime(item['date']): int(item['shares_outstanding_diluted']) if item['shares_outstanding_diluted'] is not None else None
                    for item in response.json()['data']
                },
                'basic_shares': {
                    pd.to_datetime(item['date']): int(item['shares_outstanding_basic']) if item['shares_outstanding_basic'] is not None else None
                    for item in response.json()['data']
                }
            }
        )
        shares.index.name = 'date'
        shares.index = pd.MultiIndex.from_arrays([
            [ticker]*len(shares), pd.to_datetime(shares.index)], names=['ticker', 'date'])
        shares = shares.infer_objects()

        # print(shares)
        # fix outliers
        Z = (np.log(shares['diluted_shares']) - np.log(shares['diluted_shares']).mean()) / np.log(shares['diluted_shares']).std()
        # pd.options.display.max_rows = None
        # print(pd.concat([shares['diluted_shares'], Z], axis=1))
        for z in Z[Z.abs() > 4].index:
            print(f"Fixing outlier for {ticker} on {z}")
            # Scale by power of 10 to place inbetween neighbors
            if z != shares.index[0] and z != shares.index[-1]:
                prev_share = shares.loc[shares.index[shares.index.get_loc(z) - 1], 'diluted_shares']
                next_share = shares.loc[shares.index[shares.index.get_loc(z) + 1], 'diluted_shares']
                this_share = shares.at[z, 'diluted_shares']
                if (
                    not pd.isna(prev_share) and prev_share > 0 and
                    not pd.isna(next_share) and next_share > 0 and
                    not pd.isna(this_share) and this_share > 0
                ):
                    power = int(np.round(
                        (np.log10(prev_share) + np.log10(next_share)) / 2 - np.log10(this_share)
                    ))
                    # adjust if monotonic and large enough outlier
                    if abs(power) >= 2 and (prev_share < this_share * (10 ** power) < next_share or next_share < this_share * (10 ** power) < prev_share):
                        shares.at[z, 'diluted_shares'] = this_share * (10 ** power)
                        print(f"Power Adjusted shares from {this_share} to {shares.at[z, 'diluted_shares']}")
                    elif abs(power) < 2:
                        print(f"Detected small outlier {prev_share} -> {this_share} -> {next_share}, replacing with nan")
                        shares.at[z, 'diluted_shares'] = np.nan
                    else:
                        # try geometric mean
                        shares.at[z, 'diluted_shares'] = int(np.sqrt(prev_share * next_share))
                        print(f"GM Adjusted shares from {this_share} to {shares.at[z, 'diluted_shares']}")
        shares.to_csv(f"../AlphaVantageData/Shares/{ticker}.csv")
        return shares, 1

def getAlphaVantageData(nTickers = None):
    import pandas as pd
    import time
    import os

    # AVKEY = "EVECVIOQ5AGUUTK9"
    AVKEY = os.getenv('AlphaVantage_API_KEY')

    tickers = pd.read_json("./tickers.json").T
    
    BanList = [
        # Major ETFs
        'SPY', 'QQQ', 'DIA', 'MDY', 'GBTC', 'FER',
        
        # Commodity/Metal ETFs and Trusts
        'GLD', 'IAU', 'SLV', 'PHYS', 'PSLV', 'USO',
        
        # Government Conservatorship (no public financials)
        'FNMA', 'FMCC',
        
        # Funds/CEFs
        'PDI', 'CEF',
        
        # Problematic/Incomplete Data
        'UTX', 'QDMI', 'KLAR', 'BULL', 'UELMO', 'ABTC', 'SOBO', 'HAPVD', 'RAL'
    ]
    BanList = ['CON']

    tickers = tickers[~((tickers['ticker'].str.len() == 5) & 
                    (tickers['ticker'].str[-1].isin(['Y', 'F']))) &
                  ~(tickers['ticker'].isin(BanList)) &
                  ~tickers['title'].str.contains('ETF|TRUST|FUND|INDEX', case=False, na=False)]
    if nTickers is None:
        nTickers = len(tickers)

    totalData = pd.DataFrame()
    totalShares = pd.DataFrame()
    QuereiesUsed = 0

    if os.path.exists(f"../AlphaVantageData/Finished/FinishedData.h5") and \
       os.path.exists('../AlphaVantageData/Shares/SharesData.h5'):
        print("Loading cached data...")
        totalData = pd.read_hdf('../AlphaVantageData/Finished/FinishedData.h5')
        totalShares = pd.read_hdf('../AlphaVantageData/Shares/SharesData.h5')

        totalData = totalData.sort_index(level=['Ticker', 'Reported Date'])
        totalShares = totalShares.sort_index(level=['ticker', 'date'])

        tickers = list(set(totalData.index.get_level_values('Ticker').unique().to_list()) & 
                       set(totalShares.index.get_level_values('ticker').unique().to_list()))
        tickers = pd.Series(totalData.index.get_level_values('Ticker').unique().intersection(
            totalShares.index.get_level_values('ticker').unique()
        ).to_list())
        tickers.name = 'ticker'
        return totalData, totalShares, tickers

    for ticker in tickers['ticker'].values[:nTickers]:
        # print(f"Fetching data for {ticker}...")

        if len(ticker) >= 5 and ticker.endswith('Y'):
            print(f"{ticker} looks like an ADR/ETF, skipping...")
            continue
        
        if ticker in BanList:
            print(f"{ticker} is in the ban list, skipping...")
            continue

        tData, qT = FetchTickerData(ticker, AVKEY)
        if tData.empty:
            print(f"No data fetched for {ticker}, skipping...")
            continue
        totalData = pd.concat([totalData, tData])

        tShares, qS = FetchShareData(ticker, AVKEY)
        if tShares.empty:
            print(f"No share data fetched for {ticker}, skipping...")
            continue
        totalShares = pd.concat([totalShares, tShares])
        QuereiesUsed += qT + qS


        if QuereiesUsed != 0:
            print(f"Completed fetching data for {ticker}. Total Queries Used: {QuereiesUsed}")
        if QuereiesUsed >= 72:
            print("Reached Alpha Vantage query limit for the minute, Sleeping for 60 seconds...")
            progressSleep(60)
            QuereiesUsed = 0
    print(f"Total Queries Used: {QuereiesUsed}")

    # cast all number like columns to numeric
    for col in totalData.columns:
        print(f"Column {col} type: {totalData[col].dtype}")

    tickers = list(set(totalData.index.get_level_values('Ticker').unique().to_list()) & 
                       set(totalShares.index.get_level_values('ticker').unique().to_list()))
    print(tickers)
    return totalData, totalShares, tickers




def progressSleep(seconds):
    import time
    max_width = 40
    
    for i in range(seconds, 0, -1):
        filled = int(max_width * (seconds - i) / seconds)
        bar = '#' * filled + '-' * (max_width - filled)
        print(f"\rCooldown: [{bar}] {i}s remaining   ", end='', flush=True)
        time.sleep(1)
    
    # Final message
    bar = '#' * max_width
    print(f"\rCooldown: [{bar}] Complete!        ")

def printV(msg, verbose):
    if verbose:
        print(msg)

def getNetIncome(df, verbose=False):
    printV("Getting Net Income", verbose)
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_colwidth', None)

    NI = df[(df['concept'] == 'us-gaap:NetIncomeLoss')]
    if len(NI) == 0:
        NI = df[(df['concept'] == 'us-gaap:ProfitLoss')]
    if len(NI) == 0:
        raise Warning("No Net Income or Profit Loss data found")
        return None
    NI = NI[(NI['period_end'] == NI['period_end'].max()) & (NI['period_start'] == NI['period_start'].max())]
    # print(NI[['context_ref','concept','numeric_value', 'Period Days', 'period_start', 'period_end']])
    
    if NI['numeric_value'].nunique() == 1:
        return NI['numeric_value'].iloc[0]
    else:
        NI = NI[NI.duplicated(subset=['context_ref'], keep=False)]
        return NI['numeric_value'].iloc[0]


def getRevenue(df, verbose=False):
    printV("Getting Revenue", verbose)
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_colwidth', None)

    revenue = df[(df['concept'] == 'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax')]
    # print(df['concept'].unique())
    if len(revenue) == 0:
        revenue = df[df['concept'] == 'us-gaap:Revenues']
        revenue = revenue[revenue['period_end'] == revenue['period_end'].max()]
        revenue = revenue[revenue['period_start'] == revenue['period_start'].max()].sort_values(by='context_ref', ascending=False)

        if len(revenue[revenue['context_ref'].str.len() == revenue['context_ref'].str.len().min()]['context_ref'].iloc[0]) > 10:
            if max(revenue['numeric_value']) == revenue['numeric_value'].mode().iloc[0]:
                return max(revenue['numeric_value'])
        elif max(revenue['numeric_value']) == revenue[revenue['context_ref'].str.len() == revenue['context_ref'].str.len().min()]['numeric_value'].iloc[0]:
            if max(revenue['numeric_value']) == revenue['numeric_value'].mode().iloc[0]:
                return max(revenue['numeric_value'])
            

        raise Warning("No Revenue data found")
        return None
    printV(revenue[['context_ref','concept','numeric_value', 'Period Days', 'period_start', 'period_end']], verbose)
    if revenue['numeric_value'].nunique() == 1:
        return revenue['numeric_value'].iloc[0]
    else:
        return revenue['numeric_value'].max()
    
def getShares(df, verbose=False):
    printV("Getting Shares", verbose)
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_colwidth', None)

    shares = df[(df['concept'] == 'us-gaap:WeightedAverageNumberOfSharesOutstandingBasic')]
    if len(shares) == 0:
        raise Warning("No Shares data found")
        return None
    shares = shares[(shares['period_end'] == shares['period_end'].max()) & (shares['period_start'] == shares['period_start'].max())]
    # print(shares[['concept','numeric_value', 'Period Days', 'period_start', 'period_end']])
    if shares['numeric_value'].nunique() == 1:
        return shares['numeric_value'].iloc[0]
    else:
        shares = shares[shares.duplicated(subset=['context_ref'], keep=False)]
        return shares['numeric_value'].iloc[0]

def getAssets(df, verbose=False):
    printV("Getting Assets", verbose)
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_colwidth', None)

    assets = df[(df['concept'] == 'us-gaap_Assets')]
    if len(assets) == 0:
        raise Warning("No Assets data found")
        return None
    period_end = pd.to_datetime(df.columns, format='%Y-%m-%d', errors='coerce').max().strftime('%Y-%m-%d')

    assets = assets[period_end]
    if assets.nunique() == 1:
        return assets.iloc[0]
    else:
        raise Warning("Multiple Assets values found")
        return None

def getLiabilities(df, verbose=False):
    printV("Getting Liabilities", verbose)
    import pandas as pd
    import numpy as np
    period_end = pd.to_datetime(df.columns, format='%Y-%m-%d', errors='coerce').max().strftime('%Y-%m-%d')

    liabilities = df[(df['concept'] == 'us-gaap_Liabilities')]
    printV(liabilities, verbose)
    if len(liabilities) == 0 or liabilities[period_end].isnull().any() == True:
        current = df[(df['concept'] == 'us-gaap_LiabilitiesCurrent')]
        noncurrent = df[(df['concept'] == 'us-gaap_LiabilitiesNoncurrent')]
        if len(current) == 0 and len(noncurrent) == 0:
            raise Warning("No Liabilities data found")
            return None

        liabilities = current[period_end] + noncurrent[period_end]
    else:
        liabilities = liabilities[period_end]


    if liabilities.nunique() == 1:
        return liabilities.iloc[0]
    else:
        raise Warning("Multiple Liabilities values found")
        return None
    

def getEquity(df, verbose=False):
    printV("Getting Equity", verbose)
    import pandas as pd
    import numpy as np
    pd.set_option('display.max_colwidth', None)

    equity = df[(df['concept'] == 'us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')]
    if len(equity) == 0:
        print("No Equity data found")
        return None
    period_end = pd.to_datetime(df.columns, format='%Y-%m-%d', errors='coerce').max().strftime('%Y-%m-%d')

    equity = equity[period_end]
    if equity.nunique() == 1:
        return equity.iloc[0]
    else:
        print("Multiple Equity values found")
        return None


def parseSECData():
    import edgar
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime

    tickers = pd.read_json("./tickers.json").T


    edgar.set_identity("Adam Danklefsen adamdanklefsen@gmail.com")
    etf_keywords = ['ETF', 'FUND', 'TRUST', 'INDEX', 'SHARES']

    filings_df = pd.DataFrame()
    for ticker in tickers['ticker'].values[:10]:
        if os.path.exists(f"../SEC_Filings/{ticker}.csv"):
            entry_df = pd.read_csv(f"../SEC_Filings/{ticker}.csv", index_col=['Ticker', 'Filing Date'], parse_dates=['Filing Date', 'Period End'])
            filings_df = pd.concat([filings_df, entry_df])
            continue
        entry_df = pd.DataFrame()
        c = edgar.Company(ticker)
        ETF = False
        if any(keyword in c.name.upper() for keyword in etf_keywords):
            ETF = True
        range = "2020-01-01:"+pd.Timestamp.today().strftime('%Y-%m-%d')
        filings = c.get_filings(form=["10-K", "10-Q"], filing_date=range)
        print(f"Found {len(filings)} filings for {ticker}: {c.name}")
        if len(filings) < 5:
            print(f"Only {len(filings)} filings found for {ticker}, skipping")
            continue
        if abs((filings[0].filing_date - datetime.today().date()).days) > 120:
            print(f"No recent filings found for {ticker}")
            continue

        entry_df = pd.DataFrame()
        for filing in filings:
            # print(filings[-2].xbrl().statements.income_statement())
            facts_df = filing.xbrl().facts.to_dataframe()
            # return facts_df
            # print(facts_df[facts_df['concept'].str.startswith('us-gaap')]['concept'].unique())
            Quarter = facts_df[facts_df['concept'] == 'dei:DocumentFiscalPeriodFocus']['value'].iloc[0]
            Year = facts_df[facts_df['concept'] == 'dei:DocumentFiscalYearFocus']['value'].iloc[0]
            period_end = filing.xbrl().period_of_report
            print(f"{ticker}: {filing.company} filing on {filing.filing_date} for {Quarter} {Year}")
            facts_df = facts_df[facts_df.columns[~facts_df.columns.str.startswith('dim_')]]
            facts_df = facts_df.drop(columns=['entity_identifier', 'entity_scheme', 'original_label', 'statement_type', 'statement_role'])\
                .map(lambda x: x[:100] if isinstance(x, str) and len(x) > 100 else x)\
                .map(lambda x: None if isinstance(x, str) and ('<' in x and '>' in x) else x)
            facts_df['Period Days'] = (pd.to_datetime(facts_df['period_end']) - pd.to_datetime(facts_df['period_start'])).dt.days
            if filing.form == "10-Q":
                quarterly_facts_df = facts_df[(facts_df['Period Days'] > 80) & (facts_df['Period Days'] < 100)] # Just Quarterly data - Missing instant period end data
                YTD_facts_df = facts_df[(facts_df['Period Days'] > 170) & (facts_df['Period Days'] < 350)] # Just YTD data - Missing instant period end data
                bs = filing.xbrl().statements.balance_sheet().to_dataframe()

                # return quarterly_facts_df
                # return quarterly_facts_df
                try:
                    revenue = getRevenue(quarterly_facts_df, verbose=True)
                    shares = getShares(quarterly_facts_df, verbose=True)
                    netIncome = getNetIncome(quarterly_facts_df, verbose=True)
                except:
                    return facts_df
                try:
                    assets = getAssets(bs, verbose=True)
                    liabilities = getLiabilities(bs, verbose=True)
                    equity = getEquity(bs, verbose=True)
                except:
                    return bs

                if Quarter == 'Q1':
                    YTD_revenue = revenue
                    YTD_netIncome = netIncome
                else:
                    YTD_revenue = getRevenue(YTD_facts_df)
                    YTD_netIncome = getNetIncome(YTD_facts_df)

                entry_df = pd.concat([entry_df, pd.DataFrame({
                    'Company': filing.company,
                    'ETF': ETF,
                    'SIC Code': c.sic,
                    'Industry': c.industry,
                    'Period End': period_end,
                    'Quarter': Quarter,
                    'Year': Year,
                    'Form': filing.form,
                    'Revenue': revenue,
                    'Revenue YTD': YTD_revenue,
                    'Shares': shares,
                    'Net Income': netIncome,
                    'Net Income YTD': YTD_netIncome,
                    'Total Assets': assets,
                    'Total Liabilities': liabilities,
                    'Total Equity': equity,
                }, index=pd.MultiIndex.from_tuples( [(ticker, pd.Timestamp(filing.filing_date))], names=['Ticker', 'Filing Date'] ))])
                if entry_df.iloc[-1].isnull().any():
                    print(f"Missing data in 10-Q filing for {filing.company} on {filing.filing_date}: "
                           f"Revenue entries: {revenue}, Revenue YTD entries: {YTD_revenue}, "
                           f"Shares entries: {shares}, "
                           f"Net Income entries: {netIncome}, Net Income YTD entries: {YTD_netIncome}, "
                           f"Assets: {entry_df.iloc[-1]['Total Assets']}, Liabilities: {entry_df.iloc[-1]['Total Liabilities']}, Equity: {entry_df.iloc[-1]['Total Equity']}")
                    # return facts_df

            elif filing.form == "10-K":
                facts_df = facts_df[(facts_df['Period Days'] > 350) & (facts_df['Period Days'] < 380)]
                bs = filing.xbrl().statements.balance_sheet().to_dataframe()

                try:
                    YTD_revenue = getRevenue(facts_df)
                    YTD_netIncome = getNetIncome(facts_df)
                    shares = getShares(facts_df)
                except:
                    return facts_df
                

                revenue = np.nan
                netIncome = np.nan
                
                entry_df = pd.concat([entry_df, pd.DataFrame({
                    'Company': filing.company,
                    'ETF': ETF,
                    'SIC Code': c.sic,
                    'Industry': c.industry,
                    'Period End': period_end,
                    'Quarter': Quarter,
                    'Year': Year,
                    'Form': filing.form,
                    'Revenue': revenue,
                    'Revenue YTD': YTD_revenue,
                    'Shares': shares,
                    'Net Income': netIncome,
                    'Net Income YTD': YTD_netIncome,
                    'Total Assets': getAssets(bs),
                    'Total Liabilities': getLiabilities(bs),
                    'Total Equity': getEquity(bs),
                }, index=pd.MultiIndex.from_tuples( [(ticker, pd.Timestamp(filing.filing_date))], names=['Ticker', 'Filing Date'] ))])
                if entry_df.iloc[-1].isnull().any():
                    print(f"Missing data in 10-K filing for {filing.company} on {filing.filing_date}: "
                           f"Revenue entries: {revenue}, Revenue YTD entries: {YTD_revenue}, "
                           f"Shares entries: {shares}, "
                           f"Net Income entries: {netIncome}, Net Income YTD entries: {YTD_netIncome}, "
                           f"Assets: {entry_df.iloc[-1]['Total Assets']}, Liabilities: {entry_df.iloc[-1]['Total Liabilities']}, Equity: {entry_df.iloc[-1]['Total Equity']}")
                    # return facts_df
                # if len(revenue) == 0 or len(shares) == 0 or len(netIncome) == 0:
                #     print(f"Missing data in 10-K filing for {filing.company} on {filing.filing_date}: Revenue entries: {revenue}, Shares entries: {shares}, Net Income entries: {netIncome}")
                #     return facts_df
                
            else:
                print(f"Unhandled form type: {filing.form}")
            

            print(f"{filing.company} on {filing.filing_date} for {Quarter} {Year}, ending on {period_end}:\tRevenue: {revenue}\tYTD Revenue: {YTD_revenue}\tShares: {shares}\tNet Income: {netIncome}\tYTD Net Income: {YTD_netIncome}")
        
        print(entry_df.loc[ticker])
        # Fill Equity if missing
        print("Filling missing Equity data")
        for idx in entry_df.loc[ticker][entry_df.loc[ticker]['Total Equity'].isnull()].index:
            # print(idx)
            if entry_df.loc[(ticker, idx), 'Total Assets'] is not None and \
                not pd.isna(entry_df.loc[(ticker, idx), 'Total Assets']) and \
                entry_df.loc[(ticker, idx), 'Total Liabilities'] is not None and \
                not pd.isna(entry_df.loc[(ticker, idx), 'Total Liabilities']):
                entry_df.at[(ticker, idx), 'Total Equity'] = (entry_df.loc[(ticker, idx), 'Total Assets'] - entry_df.loc[(ticker, idx), 'Total Liabilities'])
        # Fill Q4 Data for 10-K filings
        print("Filling Q4 Data for 10-K filings")
        MissingRevenues = entry_df[(entry_df.index.get_level_values(0) == ticker) & 
                           (entry_df['Quarter'] == 'FY') & 
                           (entry_df['Revenue'].isnull())].index
        # print(f"Found {len(MissingRevenues)} missing Revenue entries in 10-K filings for {ticker}")
        # print(MissingRevenues)
        for idx in MissingRevenues:
            print(idx)
            if entry_df.loc[idx, 'Revenue YTD'] is None or pd.isna(entry_df.loc[idx, 'Revenue YTD']):
                print(f"Missing Revenue YTD for {idx}, cannot fill Revenue")
                continue
            else:
                Q3_idx = entry_df.loc[ticker][(entry_df.loc[ticker]['Year'] == entry_df.loc[idx, 'Year']) & (entry_df.loc[ticker]['Quarter'] == 'Q3')].index
                if len(Q3_idx) != 1:
                    print(f"Could not find unique Q3 entry for {idx}, cannot fill Revenue")
                    continue
                if entry_df.loc[(ticker, Q3_idx[0]), 'Revenue YTD'] is None or pd.isna(entry_df.loc[(ticker, Q3_idx[0]), 'Revenue YTD']):
                    print(f"Missing Q3 Revenue YTD for {Q3_idx[0]}, cannot fill Revenue")
                    continue
                Q3RevenueYTD = entry_df.loc[(ticker, Q3_idx[0]), 'Revenue YTD']
                entry_df.at[idx, 'Revenue'] = entry_df.loc[idx, 'Revenue YTD'] - Q3RevenueYTD

                Q3NetIncomeYTD = entry_df.loc[(ticker, Q3_idx[0]), 'Net Income YTD']
                entry_df.at[idx, 'Net Income'] = entry_df.loc[idx, 'Net Income YTD'] - Q3NetIncomeYTD
            
        entry_df.to_csv(f"../SEC_Filings/{ticker}.csv")
        filings_df = pd.concat([filings_df, entry_df])
        # return filings_df
 
    return filings_df
    all_filings = edgar.get_filings(form="10-Q", filing_date="2020-01-01:2020-03-31")
    all_filings = [all_filings[0], all_filings[1]]

    # print(type(all_filings))
    # print(all_filings)

    filings_list = []
    shares = pd.Series()
    for filing in all_filings:
        xbrl = filing.xbrl()

        s = xbrl.facts.get_facts_by_concept('CommonStockSharesOutstanding')[['period_instant','value']]
        
        try:
            ticker = CIK2Ticker.loc[str(filing.cik)].upper()
            s = pd.Series(s['value'].values, index=pd.MultiIndex.from_product([[ticker], s['period_instant']], names=['Ticker','Date']))
            shares = pd.concat([shares, s])
        except Exception as e:
            ticker = None
        
        
        print(xbrl.entity_info) # BIG

        bs = xbrl.statements.balance_sheet().to_dataframe().set_index('concept')
        print(bs)

        inc = xbrl.statements.income_statement().to_dataframe().set_index('concept')
        print(inc)
        return inc
        PeriodEndDate = pd.to_datetime(bs.columns, format='%Y-%m-%d', errors='coerce').max().strftime('%Y-%m-%d')
        print(f"Period End Date: {PeriodEndDate}")

        if 'us-gaap_Liabilities' in bs.index:
            Debt = bs.loc['us-gaap_Liabilities', PeriodEndDate]
        else:
            if 'us-gaap_LiabilitiesCurrent' in bs.index and 'us-gaap_LiabilitiesNoncurrent' in bs.index:
                Debt = bs.loc['us-gaap_LiabilitiesCurrent', PeriodEndDate] + bs.loc['us-gaap_LiabilitiesNoncurrent', PeriodEndDate]

        filings_list.append({
            'ticker': ticker,
            'filing_date': filing.filing_date,
            'company': filing.company,
            'cik': filing.cik,
            'form': filing.form,
            'accession_number': filing.accession_number,
            'Total Assets': bs.loc['us-gaap_Assets', PeriodEndDate] if 'us-gaap_Assets' in bs.index else None,
            'Total Liabilities': Debt,
            'Total Equity': bs.loc['us-gaap_StockholdersEquity', PeriodEndDate] if 'us-gaap_StockholdersEquity' in bs.index else None,
            'Total Revenue': inc.loc['us-gaap_Revenues', PeriodEndDate] if 'us-gaap_Revenues' in inc.index else None,
            'Net Income': inc.loc['us-gaap_NetIncomeLoss', PeriodEndDate] if 'us-gaap_NetIncomeLoss' in inc.index else None,
        })
        print(filings_list[-1])
    print(filings_list)
    filings_df = pd.DataFrame(filings_list)
    filings_df = filings_df.set_index(['ticker', 'filing_date'])

    return filings_df