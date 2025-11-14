def get_all_tickers(cache_file: str = None):
    import simfin as sf
    import os
    import yfinance as yf
    import pandas as pd
    import numpy as np

    if cache_file is not None and os.path.exists(cache_file):
        d = pd.read_csv(cache_file)
        companies = d['Ticker'].tolist()
        print(f"Loaded {len(companies)} tickers from cache file {cache_file}.")
        return companies, d

    # Pull Balance Sheet Data
    api_key = os.getenv('SIMFIN_API_KEY')
    sf.config.set_api_key(api_key=api_key)
    sf.config.set_data_dir('../simfin_data/')
    companies = sf.load_companies(market='us')
    companies = companies.index.get_level_values('Ticker').unique().dropna().tolist()
    # companies = [comp for comp in companies if not comp.endswith('_delisted')]
    companies = [comp for comp in companies if len(comp) <=7]
    companies = [comp for comp in companies if not '_' in comp]

    # companies = companies  # Limit to first 500 for speed

    T = yf.Tickers(companies)
    d = T.download(period="1d").iloc[-1]['Close'].T.fillna(value=pd.NA)
    # print(d)
    d = d.dropna()
    companies = pd.Series(d.dropna().index)
    # companies = [comp for comp in companies if not d[comp] == np.nan]
        #  or yf.Ticker(comp).info['exchange'] is None or yf.Ticker(comp).info['tradeable'] is False
    pd.DataFrame(companies, columns=["Ticker"]).to_csv("tickers.csv", index=False)

    print(f"Found {len(companies)} valid tickers out of {len(d)} total tickers.")

    return companies, d
# Pulls balance sheet data and calculates B2M for all Assets or given subset
def Book_to_Market_Ratio_Multi_Asset(assets: list[str] = None):
    import simfin as sf
    import yfinance as yf
    import pandas as pd
    import os

    # Pull Balance Sheet Data
    api_key = os.getenv('SIMFIN_API_KEY')
    sf.config.set_api_key(api_key=api_key)
    sf.config.set_data_dir('../simfin_data/')

    balance_normal = sf.load_balance(variant='quarterly', market='us')
    balance_banks = sf.load_balance_banks(variant='quarterly', market='us')
    balance_insurance = sf.load_balance_insurance(variant='quarterly', market='us')
    balance_sheet = pd.concat([balance_normal, balance_banks, balance_insurance])

    income_normal = sf.load_income(variant='quarterly', market='us')
    income_banks = sf.load_income_banks(variant='quarterly', market='us')
    income_insurance = sf.load_income_insurance(variant='quarterly', market='us')
    income_sheet = pd.concat([income_normal, income_banks, income_insurance])


    if assets is not None:
        # assets not in balance sheet
        balance_sheet_assets = balance_sheet.index.get_level_values('Ticker').unique().tolist()
        print(balance_sheet_assets)
        missing_assets = [asset for asset in assets if asset not in balance_sheet_assets]
        print(f"""Warning: The following {len(missing_assets)} / {len(assets)} assets are missing from the balance sheet data and will be skipped: {missing_assets}""")
        assets = [asset for asset in assets if asset in balance_sheet_assets]
        assets = [asset for asset in assets if not asset.endswith('_delisted')]
        balance_sheet = balance_sheet.loc[assets]


        income_sheet_assets = income_sheet.index.get_level_values('Ticker').unique().tolist()
        missing_assets = [asset for asset in assets if asset not in income_sheet_assets]
        print(f"""Warning: The following {len(missing_assets)} / {len(assets)} assets are missing from the income sheet data and will be skipped: {missing_assets}""")
        assets = [asset for asset in assets if asset in income_sheet_assets]
        income_sheet = income_sheet.loc[assets]
    else:
        assets = balance_sheet.index.get_level_values('Ticker').unique().tolist()


    print(f"Found Data for {len(assets)} assets.")

    yf_ticker = yf.Tickers(' '.join(assets))
    balance_sheet = balance_sheet[['Fiscal Year','Fiscal Period','Publish Date','Restated Date','Shares (Basic)','Total Assets','Total Liabilities', 'Total Equity']]
    startdate = balance_sheet.index.get_level_values('Report Date').min()

  
    

    price_data = cached_yf_price_data_download(yf_ticker, start_date=startdate, cache_hdf5='../simfin_data/yf_price_cache/price_data.h5')
    info = pd.DataFrame(0, columns=assets, index=['industry','sector','bookValue','priceToBook','currentRatio'])

    selected_fields = ['Share Issued','Total Assets','Total Liabilities Net Minority Interest','Stockholders Equity']
    info_fields = ['displayName','industry','sector','bookToPrice',
                   'epmcTrailingTwelveMonths', 'profitMargins', 'revenuePerMC', 'freeCashFlowYield', 'EBITDAonEV',
                   'returnOnAssets', 'returnOnEquity', 'equityToDebt', 'quickRatio', 'currentRatio', 'revenueGrowth',
                   'earningsGrowth','earningsQuarterlyGrowth', 'dividendYield', 'payoutRatio', 'institutionalVsInsider']
                    # 'operatingCashFlow', 'freeCashFlow'

    non_numeric_info_fields = ['displayName','industry','sector', 'recommendationMean', 'numberOfAnalystOpinions']

    raw_features = [
                    'profitMargins',                # Net Income / Revenue	                                Higher = Better	    Ratio
                    'returnOnAssets',               # Net Income / Total Assets	                            Higher = Better	    Ratio
                    'returnOnEquity',               # Net Income / Equity	                                Higher = Better	    Ratio
                    'revenueGrowth',                # YoY Revenue Growth	                                Higher = Better	    Ratio
                    'earningsGrowth',               # YoY Earnings Growth	                                Higher = Better	    Ratio
                    'currentRatio',                 # Current Assets / Current Liabilities	                Higher = Better	    Ratio
                    'quickRatio',                   # (Current Assets - Inventory) / Current Liabilities	Higher = Better	    Ratio
                    'trailingPegRatio',             # PE / Growth Rate	                                    Uncorrelated	    Ratio
                    'beta',                         # Volatility vs. S&P 500

                    'operatingMargins',	            # Operating Income / Revenue	                        Higher = Better	    Ratio
                    'grossMargins',	                # Gross Profit / Revenue	                            Higher = Better	    Ratio
                    'earningsQuarterlyGrowth',	    # QoQ Earnings Growth	                                Higher = Better	    Ratio
                    'dividendYield',	            # Annual Dividend / Price	                            Uncorrelated	    Ratio
                    'payoutRatio',	                # Dividends / Earnings	                                Uncorrelated	    Ratio
                    'heldPercentInstitutions',	    # % held by institutions	                            Uncorrelated	    Percentage
                    'heldPercentInsiders',	        # % held by insiders	                                Uncorrelated	    Percentage
                    'shortPercentOfFloat',	        # Short interest / Float	                            Uncorrelated	    Percentage
                    'enterpriseToRevenue',	        # EV / Revenue	                                        Uncorrelated	    Ratio
    ]
    inversion_features = [
                    'bookToPrice',                  # 1 / Price-to-Book                                     Higher = Better     Ratio
                    'equityToDebt',                 # 1 / Debt-to-Equity                                    Higher = Better     Ratio

                    'salesToPrice',                 # 1 / Price-to-Sales                                    Higher = Better     Ratio
                    'forwardEarningsToPrice',       # 1 / Forward PE                                        Higher = Better     Ratio
    ]

    calculated_features = [
                    'freeCashFlowYield',            # FCF / Market Cap		                                Higher = Better	    Ratio
                    'EBITDAonEV',                   # EBITDA / Enterprise Value		                        Higher = Better	    Ratio
                    'impliedUpside',                # (Target - Price) / Price		                        Higher = Better	    Ratio

                    'epmcTrailingTwelveMonths',     # EPS / Price (Earnings Yield)		                    Higher = Better	    Ratio
                    'revenuePerMC',                 # Revenue per Share / Price		                        Higher = Better	    Ratio
                    'FCFtoNetIncome',               # FCF / Net Income (Quality)		                    Higher = Better	    Ratio
                    'operatingCFtoRevenue',         # Operating CF / Revenue		                        Higher = Better	    Ratio
                    'distanceFromHigh',             # (Price - 52w High) / 52w High		                    Uncorrelated	    Ratio
                    'priceVs200DayMA',              # Price / 200-day MA		                            Uncorrelated	    Ratio
                    'priceVs50DayMA',               # Price / 50-day MA		                                Uncorrelated	    Ratio
                    'netCashToMC',                  # (Cash - Debt) / Market Cap		                    Higher = Better	    Ratio
                    'forwardVsTrailingPE',          # Forward PE / Trailing PE		                        Uncorrelated	    Ratio
    ]

    bal_yf_full = cached_yf_balance_sheet_download(yf_ticker, cache_hdf5='../simfin_data/yf_balance_cache/balance_sheet.h5')
    income_yf_full = cached_yf_income_sheet_download(yf_ticker, cache_hdf5='../simfin_data/yf_income_cache/income_sheet.h5')
    earnings_dates_yf_full = cached_yf_earnings_dates_download(yf_ticker, cache_hdf5='../simfin_data/yf_earnings_dates_cache/earnings_dates.h5')

    bal_yf = bal_yf_full.get(selected_fields)

    bal_yf['Fiscal Year'] = pd.Series(bal_yf.index.get_level_values('Report Date').to_period('Y').year.values, index=bal_yf.index)
    bal_yf['Fiscal Period'] = pd.Series('Q' + bal_yf.index.get_level_values('Report Date').to_period('Q').quarter.astype(str).values, index=bal_yf.index)
    bal_yf = bal_yf[['Fiscal Year','Fiscal Period','Share Issued','Total Assets','Total Liabilities Net Minority Interest','Stockholders Equity']]
    bal_yf.columns = ['Fiscal Year','Fiscal Period','Shares','Assets','Liabilities','Equity']
    balance_sheet.columns = ['Fiscal Year','Fiscal Period','Publish','Restated','Shares','Assets','Liabilities','Equity']


    bal = pd.concat([balance_sheet, bal_yf])
    bal = bal.dropna(subset=['Shares'])
    bal = bal.loc[~bal.index.duplicated(keep='first')]
    bal = bal.sort_index(level=['Ticker','Report Date'], ascending=[True,True])

    

    
    # replace missing publish dates in bal with data from earnings_dates_yf_full
    missing_publish_dates = bal[bal['Publish'].isna()].index
    earnings_series = earnings_dates_yf_full['Earnings Date']
    # return 0,0,0,0,0, earnings_series
    print("Earnings Series:")
    print(earnings_series)
    print("Missing Publish Dates:")
    print(missing_publish_dates)
    common_missing = missing_publish_dates.intersection(earnings_series.index)
    print("Common Missing Publish Dates:")
    print(common_missing)
    print(f"{len(common_missing)} / {len(missing_publish_dates)} missing publish dates have earnings date data available.")
    print("Earnings series at missing publish dates:")
    print(earnings_series.loc[common_missing])
    print("Duplicate indicies:")
    print(earnings_series.index[earnings_series.index.duplicated(keep=False)])
    print("Bal at common missing publish dates before fill:")
    print(bal.loc[common_missing, 'Publish'])
    print("Data Compare:")
    print(pd.concat([bal.loc[common_missing, 'Publish'], earnings_series.loc[common_missing]], axis=1))
    bal.loc[common_missing, 'Publish'] = earnings_series.loc[common_missing]

    Most_Recent_Quarter_end = pd.Timestamp.today() - pd.offsets.QuarterEnd()
    print(f"Most Recent Quarter End: {Most_Recent_Quarter_end.date()}")
    print("Entries in Earnings Series not in Bal:")
    earnings_not_in_bal = earnings_series.index.difference(bal.index)
    # earnings_not_in_bal = earnings_not_in_bal[earnings_not_in_bal.get_level_values('Report Date').normalize() != Most_Recent_Quarter_end.normalize()]
    print(earnings_not_in_bal)

    missing_publish_dates = bal[bal['Publish'].isna()].index
    print("Missing Publish Dates:")
    print(missing_publish_dates)

    for ticker in missing_publish_dates.get_level_values('Ticker').unique():
        # print(f"Processing missing publish dates for {ticker}...")
        if ticker not in earnings_series.index.get_level_values('Ticker'):
            # print(f"  No earnings data available for {ticker}, skipping...")
            continue
        ticker_missing_dates = missing_publish_dates[missing_publish_dates.get_level_values('Ticker') == ticker].get_level_values('Report Date')
        ticker_earnings = earnings_series.loc[ticker].index
        for bal_date in ticker_missing_dates:
            close_match = ticker_earnings[abs(ticker_earnings - bal_date) <= pd.Timedelta(days=5)]
            if len(close_match) > 0:
                print(f"{ticker}: {bal_date} matches {close_match[0]}")
                # bal.loc[(ticker, bal_date), 'Publish'] = earnings_series.loc[(ticker, close_match[0])]
                row_data = bal.loc[(ticker, bal_date)].copy()
                row_data['Publish'] = earnings_series.loc[(ticker, close_match[0])]
                bal = bal.drop((ticker, bal_date))
                bal.loc[(ticker, close_match[0])] = row_data




    # print("Filling missing publish dates...")
    # for idx in bal.loc[missing_publish_dates].index:
    #     if idx in earnings_dates_yf_full.index:
    #         print(idx)
    #         print(earnings_dates_yf_full.loc[idx, 'Earnings Date'])
    #         # print(f"Filling missing publish date for {idx} with {earnings_dates_yf_full.at[idx, 'Earnings Date']}")
    #         bal.at[idx, 'Publish'] = earnings_dates_yf_full.loc[idx, 'Earnings Date']


    return bal,0,0,0,0, earnings_dates_yf_full


    income_yf = income_yf_full.get(['Net Income'])
    income_yf.columns = ['Net Income']
    income_sheet = income_sheet.get(['Net Income'])
    # print(income_sheet)
    income_sheet.columns = ['Net Income']

    income = pd.concat([income_sheet, income_yf])
    income = income.dropna(subset=['Net Income'])
    income = income.loc[~income.index.duplicated(keep='first')]
    income = income.sort_index(level=['Ticker','Report Date'], ascending=[True,True])



    info = cached_yf_info_download(yf_ticker, info_fields, cache_hdf5='../simfin_data/yf_info_cache/info.h5').T
    info = info.apply(pd.to_numeric, errors='ignore')
    print(info.columns)
    info = info.loc[info['totalRevenue'] != 0]

    info_out = info[non_numeric_info_fields + raw_features].copy()

    # Inverted Features
    info_out['bookToPrice'] = 1 / info['priceToBook']
    info_out['equityToDebt'] = 1 / info['debtToEquity']
    info_out['salesToPrice'] = 1 / info['priceToSalesTrailing12Months']
    info_out['forwardEarningsToPrice'] = 1 / info['forwardPE']

    # Calculated Features
    info_out['freeCashFlowYield'] = info['freeCashflow'] / info['marketCap']
    info_out['EBITDAonEV'] = info['ebitda'] / info['enterpriseValue']
    info_out['impliedUpside'] = (info['targetMeanPrice'] - info['previousClose']) / info['previousClose']

    info_out['epmcTrailingTwelveMonths'] = info['epsTrailingTwelveMonths'] / info['previousClose']
    info_out['revenuePerMC'] = info['revenuePerShare'] / info['previousClose']
    info_out['FCFtoNetIncome'] = info['freeCashflow'] / info['netIncomeToCommon']
    info_out['operatingCFtoRevenue'] = info['operatingCashflow'] / ( info['totalRevenue'] + 1e-7 )
    info_out['distanceFromHigh'] = (info['previousClose'] - info['fiftyTwoWeekHigh']) / info['fiftyTwoWeekHigh']
    info_out['priceVs200DayMA'] = info['previousClose'] / info['twoHundredDayAverage']
    info_out['priceVs50DayMA'] = info['previousClose'] / info['fiftyDayAverage']
    info_out['netCashToMC'] = (info['totalCash'] - info['totalDebt']) / info['marketCap']
    info_out['forwardVsTrailingPE'] = info['forwardPE'] / info['trailingPE']

    info_out['recommendationMean'] = 5 - info['recommendationMean']

    
    # print(info)
    # print("Info Fields:")
    # print(info.T.columns)

    return bal, price_data.fillna(method='ffill'), info_out, info, income, earnings_dates_yf_full



def plot_balance_sheet(balance, asset: str = None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    if asset is not None:
        balance = balance[asset]

    plt.figure(figsize=(12,6))
    plt.step(balance.index,balance['Assets'].values, label='Total Assets')
    plt.step(balance.index,balance['Liabilities'].values, label='Total Liabilities')
    plt.step(balance.index,balance['Equity'].values, label='Total Equity')
    if asset is not None:
        plt.title(f'Balance Sheet for {asset}')
    plt.gca().yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f'${x/1e9:,.2f}B')
    )
    plt.legend()
    plt.show()
    return

def plot_book_to_market_ratio(book_value, market_value, asset: str = None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    if asset is not None:
        book_value = book_value[asset]
        market_value = market_value[asset]
    else:
        assets = book_value.columns.tolist()

    plt.figure(figsize=(12,6))
    if asset is None:
        for a in assets:
            plt.plot(book_value[a] / market_value[a], label=f'Book-to-Market for {a}')
    else:
        plt.plot(book_value / market_value, label='Book-to-Market')
    # ax2 = plt.gca().twinx()
    # ax2.plot(market_value / book_value, color='orange', label='Market-to-Book Ratio')
    if asset is not None:
        plt.title(f'Book-to-Market Ratio for {asset}')
    plt.gca().yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, pos: f'{x:,.2f}')
    )
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    return

def cached_yf_price_data_download(yf_ticker, start_date, cache_hdf5: str = None):
    import os
    import pandas as pd
    import tables
    print(f"Fetching price data for {len(yf_ticker.tickers)} tickers...")

    tickers_to_fetch = yf_ticker.tickers.keys()
    price_data = pd.DataFrame(0, columns=yf_ticker.tickers.keys(), index=pd.date_range(start=start_date, end=pd.Timestamp.today(), freq='B', tz='US/Eastern'))


    if cache_hdf5 is not None and os.path.exists(cache_hdf5):
        price_data = pd.read_hdf(cache_hdf5, key='price_data')

        # Check is hdf5 has all items
        missing_tickers = [ticker for ticker in yf_ticker.tickers.keys() if ticker not in price_data.columns]
        if len(missing_tickers) > 0:
            print(f"Cache file {cache_hdf5} is missing tickers: {missing_tickers}. Re-downloading all price data.")
            os.remove(cache_hdf5)
            tickers_to_fetch = yf_ticker.tickers.keys()
        else:
            print(f"Loaded price data from cache file {cache_hdf5}.")
            return price_data.round(5)
        
    max_width = 40

    cache_dir = '../simfin_data/yf_price_cache/'
    os.makedirs(cache_dir, exist_ok=True)
    for i, ticker in enumerate(yf_ticker.tickers.keys()):
        filled = int(max_width * i / len(yf_ticker.tickers))
        bar = '#' * filled + '-' * (max_width - filled)
        print(f"\rProcessing {ticker}: [{bar}] {int(100 * i / len(yf_ticker.tickers))}%  ", end='', flush=True)

        cache_file = os.path.join(cache_dir, f'{ticker}_price_data.csv')

        if os.path.exists(cache_file):
            price_data[ticker] = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            price_data[ticker] = yf_ticker.tickers[ticker].history(start=start_date)['Close']
            price_data[ticker].to_csv(cache_file)
    print("\rProcessing complete.                           ")

    price_data.to_hdf(cache_dir + 'price_data.h5', key='price_data', mode='w')
    return price_data.round(5)


def cached_yf_income_sheet_download(yf_ticker, cache_hdf5: str = None):
    import os
    import pandas as pd
    import tables

    print(f"Fetching income sheet data for {len(yf_ticker.tickers)} tickers...")

    # income_sheet = pd.DataFrame()
    income_sheet = pd.DataFrame()

    if cache_hdf5 is not None and os.path.exists(cache_hdf5):
        income_sheet = pd.read_hdf(cache_hdf5, key='income_sheet')
        print(f"Loaded income sheet data from cache file {cache_hdf5}.")
        return income_sheet
    
    max_width = 40
    cache_dir = '../simfin_data/yf_income_cache/'
    os.makedirs(cache_dir, exist_ok=True)
    for i, ticker in enumerate(yf_ticker.tickers.keys()):
        filled = int(max_width * i / len(yf_ticker.tickers))
        bar = '#' * filled + '-' * (max_width - filled)
        print(f"\rProcessing {ticker}: [{bar}] {int(100 * i / len(yf_ticker.tickers))}%    ", end='', flush=True)

        cache_file = os.path.join(cache_dir, f'{ticker}_income_sheet.csv')

        if os.path.exists(cache_file):
            inc = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            inc = yf_ticker.tickers[ticker].quarterly_income_stmt.T
            inc.to_csv(cache_file)

        inc.index = pd.MultiIndex.from_product([[ticker], inc.index], names=['Ticker', 'Report Date'])
        income_sheet = pd.concat([income_sheet, inc])
    print("\rProcessing complete.                           ")

    income_sheet.to_hdf(cache_dir + 'income_sheet.h5', key='income_sheet', mode='w')
    return income_sheet

def cached_yf_balance_sheet_download(yf_ticker, cache_hdf5: str = None):
    import os
    import pandas as pd
    import tables

    print(f"Fetching balance sheet data for {len(yf_ticker.tickers)} tickers...")

    # balance_sheet = dict()
    balance_sheet = pd.DataFrame()

    if cache_hdf5 is not None and os.path.exists(cache_hdf5):
        balance_sheet = pd.read_hdf(cache_hdf5, key='balance_sheet')
        print(f"Loaded balance sheet data from cache file {cache_hdf5}.")
        return balance_sheet

    max_width = 40

    cache_dir = '../simfin_data/yf_balance_cache/'
    os.makedirs(cache_dir, exist_ok=True)
    for i, ticker in enumerate(yf_ticker.tickers.keys()):
        filled = int(max_width * i / len(yf_ticker.tickers))
        bar = '#' * filled + '-' * (max_width - filled)
        print(f"\rProcessing {ticker}: [{bar}] {int(100 * i / len(yf_ticker.tickers))}%", end='', flush=True)

        cache_file = os.path.join(cache_dir, f'{ticker}_balance_sheet.csv')

        if os.path.exists(cache_file):
            bal = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            bal = yf_ticker.tickers[ticker].quarterly_balance_sheet.T
            bal.to_csv(cache_file)

        bal.index = pd.MultiIndex.from_product([[ticker], bal.index], names=['Ticker', 'Report Date'])
        balance_sheet = pd.concat([balance_sheet, bal])

    print("\rProcessing complete.                           ")

    balance_sheet.to_hdf(cache_dir + 'balance_sheet.h5', key='balance_sheet', mode='w')
    return balance_sheet

def cached_yf_info_download(yf_ticker, selected_keys, cache_hdf5: str = None):
    import os
    import pandas as pd
    import tables

    print(f"Fetching info data for {len(yf_ticker.tickers)} tickers...")

    info = pd.DataFrame(0, columns=yf_ticker.tickers.keys(), index=selected_keys)
    if cache_hdf5 is not None and os.path.exists(cache_hdf5):
        info = pd.read_hdf(cache_hdf5, key='info')
        print(f"Loaded info data from cache file {cache_hdf5}.")
        return info
    
    max_width = 40

    cache_dir = '../simfin_data/yf_info_cache/'
    os.makedirs(cache_dir, exist_ok=True)
    for i, ticker in enumerate(yf_ticker.tickers.keys()):
        filled = int(max_width * i / len(yf_ticker.tickers))
        bar = '#' * filled + '-' * (max_width - filled)
        print(f"\rProcessing {ticker}: [{bar}] {int(100 * i / len(yf_ticker.tickers))}%", end='', flush=True)

        cache_file = os.path.join(cache_dir, f'{ticker}_info.csv')

        if os.path.exists(cache_file):
            info[ticker] = pd.read_csv(cache_file, index_col=0, squeeze=True)
        else:
            info[ticker] = pd.Series(yf_ticker.tickers[ticker].info)
            info[ticker].to_csv(cache_file)

    print("\rProcessing complete.                           ")

    info.to_hdf(cache_dir + 'info.h5', key='info', mode='w')
    return info

# Pass in Tickers object, but need to download by singular ticker objects
def cached_yf_earnings_dates_download(yf_ticker, cache_hdf5: str = None):
    import os
    import pandas as pd
    import tables
    import yfinance as yf

    print(f"Fetching earnings dates data for {len(yf_ticker.tickers)} tickers...")
    from openbb import obb

    earnings_dates = pd.DataFrame()
    if cache_hdf5 is not None and os.path.exists(cache_hdf5):
        earnings_dates = pd.read_hdf(cache_hdf5, key='earnings_dates')
        print(f"Loaded earnings dates data from cache file {cache_hdf5}.")
        return earnings_dates
    
    max_width = 40

    cache_dir = '../simfin_data/yf_earnings_dates_cache/'
    os.makedirs(cache_dir, exist_ok=True)
    for i, ticker in enumerate(yf_ticker.tickers.keys()):
        filled = int(max_width * i / len(yf_ticker.tickers))
        bar = '#' * filled + '-' * (max_width - filled)
        print(f"\rProcessing {ticker}: [{bar}] {int(100 * i / len(yf_ticker.tickers))}%   ", end='', flush=True)

        cache_file = os.path.join(cache_dir, f'{ticker}_earnings_dates.csv')

        if os.path.exists(cache_file):
            DL = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            try:
                earnings = obb.equity.fundamental.filings(ticker, provider='sec',
                                                        form_type=["10-Q", "10-K"], limit=6).to_df()
            except Exception as e:
                print(f"OpenBB earnings dates fetch failed for {ticker} with error: {e}")
                continue
            if 'report_date' not in earnings.columns:
                print(f"OpenBB earnings dates not found for {ticker}. Skipping.")
                continue
            earnings = earnings[['report_date', 'filing_date']]
            earnings.index = pd.to_datetime(earnings['report_date'])
            earnings.index.name = 'Report Date'
            earnings = earnings.drop(columns=['report_date'])
            earnings.columns = ['Earnings Date']
            earnings = earnings[~earnings.index.duplicated(keep='last')]
            DL = earnings
            DL.to_csv(cache_file)

        DL.index = DL.index.normalize() 
        DL.index = pd.MultiIndex.from_product([[ticker], DL.index], names=['Ticker', 'Report Date'])
        earnings_dates = pd.concat([earnings_dates, DL])

    print("\rProcessing complete.                           ")

    earnings_dates.to_hdf(cache_dir + 'earnings_dates.h5', key='earnings_dates', mode='w')
    return earnings_dates