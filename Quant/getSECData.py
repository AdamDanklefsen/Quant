def FetchTickerData(ticker, apiKey):
    # Check if cached finished file exists
    import os
    import pandas as pd
    
    if os.path.exists(f"../AlphaVantageData/Finished/{ticker}.csv"):
        # print(f"Reading cached finished file for {ticker}...")
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
        for col in df.columns:
            print(f"Column {col} type: {df[col].dtype}")
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
        income = pd.DataFrame(response.json()['quarterlyReports'])
        income.set_index('fiscalDateEnding', inplace=True)
        income = income.infer_objects()
        income.to_csv(f"../AlphaVantageData/IncomeStatements/{ticker}.csv")
        return income, 1


def getAlphaVantageData(nTickers = None):
    import pandas as pd
    import time


    tickers = pd.read_json("./tickers.json").T
    AVKEY = "EVECVIOQ5AGUUTK9"
    BanList = ['SPY',
                'QQQ',
                'UTX',
                'GLD',
                'IAU',
                'GBTC',
                'FER',
                'DIA',
                'MDY']

    if nTickers is None:
        nTickers = len(tickers)

    totalData = pd.DataFrame()
    QuereiesUsed = 0
    for ticker in tickers['ticker'].values[:nTickers]:

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