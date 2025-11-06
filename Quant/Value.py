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

    if assets is not None:
        # assets not in balance sheet
        balance_sheet_assets = balance_sheet.index.get_level_values('Ticker').unique().tolist()
        print(balance_sheet_assets)
        missing_assets = [asset for asset in assets if asset not in balance_sheet_assets]
        print(f"""Warning: The following {len(missing_assets)} / {len(assets)} assets are missing from the balance sheet data and will be skipped: {missing_assets}""")
        assets = [asset for asset in assets if asset in balance_sheet_assets]
        assets = [asset for asset in assets if not asset.endswith('_delisted')]
        balance_sheet = balance_sheet.loc[assets]
    else:
        assets = balance_sheet.index.get_level_values('Ticker').unique().tolist()

    print(len(assets))
    print(assets)

    yf_ticker = yf.Tickers(' '.join(assets))
    balance_sheet = balance_sheet[['Publish Date','Restated Date','Shares (Basic)','Total Assets','Total Liabilities', 'Total Equity']]
    # print(balance_sheet)
    startdate = balance_sheet.index.get_level_values('Report Date').min()

    out = {}
    Market_Value = pd.DataFrame(0, columns=assets, index=pd.date_range(start=startdate, end=pd.Timestamp.today(), freq='B'))
    Book_Value = pd.DataFrame(0, columns=assets, index=pd.date_range(start=startdate, end=pd.Timestamp.today(), freq='B'))
    
    # price_data = pd.DataFrame(0, columns=assets, index=pd.date_range(start=startdate, end=pd.Timestamp.today(), freq='B'))
    # print(yf_ticker)
    print('Fetching price data...')
    price_data = cached_yf_price_data_download(yf_ticker, start_date=startdate, cache_hdf5='../simfin_data/yf_price_cache/price_data.h5')
    # price_data = yf_ticker.history(start=balance_sheet.index.get_level_values('Report Date')[0])['Close']
    info = pd.DataFrame(0, columns=assets, index=['industry','sector','bookValue','priceToBook','currentRatio'])
    # print(price_data.head())
    # print(price_data)

    simfin_selected_fields = ['Publish Date','Restated Date','Shares (Basic)','Total Assets','Total Liabilities', 'Total Equity']
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

    # print(bal_yf_full.get(selected_fields))
    
    # print("SimFin Balance Sheet Fields:")
    # print(balance_sheet[simfin_selected_fields])

    bal_yf = bal_yf_full.get(selected_fields)
    bal_yf.columns = ['Shares','Assets','Liabilities','Equity']
    balance_sheet = balance_sheet.get(simfin_selected_fields)
    balance_sheet.columns = ['Publish','Restated','Shares','Assets','Liabilities','Equity']

    bal = pd.concat([balance_sheet, bal_yf])
    bal = bal.dropna(subset=['Shares'])
    bal = bal.loc[~bal.index.duplicated(keep='first')]
    bal = bal.sort_index(level=['Ticker','Report Date'], ascending=[True,True])
    # print("Combined Balance Sheet Fields:")
    # print(bal)


    info = cached_yf_info_download(yf_ticker, info_fields, cache_hdf5='../simfin_data/yf_info_cache/info.h5').T
    info = info.apply(pd.to_numeric, errors='ignore')
    info = info.loc[info['totalRevenue'] != 0]

    info_out = info[non_numeric_info_fields + raw_features]

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

    return bal, price_data, info_out, info



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

    tickers_to_fetch = yf_ticker.tickers.keys()
    price_data = pd.DataFrame(0, columns=yf_ticker.tickers.keys(), index=pd.date_range(start=start_date, end=pd.Timestamp.today(), freq='B'))


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
            return price_data
        
    max_width = 40

    cache_dir = './simfin_data/yf_price_cache/'
    os.makedirs(cache_dir, exist_ok=True)
    for i, ticker in enumerate(yf_ticker.tickers.keys()):
        filled = int(max_width * i / len(yf_ticker.tickers))
        bar = '#' * filled + '-' * (max_width - filled)
        print(f"\rProcessing {ticker}: [{bar}] {int(100 * i / len(yf_ticker.tickers))}%", end='', flush=True)

        cache_file = os.path.join(cache_dir, f'{ticker}_price_data.csv')

        if os.path.exists(cache_file):
            price_data[ticker] = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            price_data[ticker] = yf_ticker.tickers[ticker].history(start=start_date)['Close']
            price_data[ticker].to_csv(cache_file)
    print("\rProcessing complete.                           ")

    price_data.to_hdf(cache_dir + 'price_data.h5', key='price_data', mode='w')
    return price_data

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

    cache_dir = './simfin_data/yf_balance_cache/'
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

    info = pd.DataFrame(0, columns=yf_ticker.tickers.keys(), index=selected_keys)
    if cache_hdf5 is not None and os.path.exists(cache_hdf5):
        info = pd.read_hdf(cache_hdf5, key='info')
        print(f"Loaded info data from cache file {cache_hdf5}.")
        return info
    
    max_width = 40

    cache_dir = './simfin_data/yf_info_cache/'
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