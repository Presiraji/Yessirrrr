

            
    
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            import matplotlib.dates as mdates
            import yfinance as yf
            import datetime
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            import os, pickle

            DD =30
            SN =22
            EMA12=13
            EMA26=39
            EMA9=9
            Ticker = "DMLP"
            start_date1 = "2023-01-01"


            df = yf.Ticker(Ticker)
            df = df.history(period="max")
            df = df.loc[start_date1:].copy()
            df.index = df.index.strftime('%Y-%m-%d')
            df.index = pd.to_datetime(df.index)
            del df['Open']
            del df['High']
            del df['Low']
            del df['Volume']
            del df['Dividends']
            del df['Stock Splits']


            # Calculating the 12 EMA
            df['EMA_19'] = df['Close'].ewm(span=EMA12, adjust=False).mean()
            df['EMA_39'] = df['Close'].ewm(span=EMA26, adjust=False).mean()
            df['EMA_9'] = df['Close'].ewm(span=EMA9, adjust=False).mean()
            df['MACD'] = df['EMA_19'] - df['EMA_39']
            # Assuming df['MACD'] is already calculated as shown in your code
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['Histogram'] = df['MACD'] - df['Signal']
            del df['EMA_19']
            del df['EMA_39']
            del df['EMA_9']


            df

            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import pandas as pd

            # Assuming 'df' is your DataFrame, and its index is of datetime type. If not, you've already converted it.
            # Also assuming 'df' already has 'MACD', 'Signal', and 'Close' columns calculated.

            # Sorting the DataFrame by the index (date) to ensure the data is in chronological order
            df = df.sort_index()

            # Calculate the Histogram if not already done
            df['Histogram'] = df['MACD'] - df['Signal']

            # Creating a figure and a grid of subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

            # Plotting the Close price on ax1
            ax1.plot(df.index, df['Close'], label='Close', color='skyblue')
            ax1.set_title('Close Price')
            ax1.legend(loc='upper left')
            ax1.grid(True)

            # Formatting the date to ensure that the date does not overlap
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=45))
            ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

            # Plotting MACD and Signal Line on ax2
            ax2.plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1.5)
            ax2.plot(df.index, df['Signal'], label='Signal Line', color='red', linewidth=1.5)

            # Plotting the Histogram as bar plot on ax2
            ax2.bar(df.index, df['Histogram'], label='Histogram', color='grey', alpha=0.3)

            ax2.set_title('MACD, Signal Line, and Histogram')
            ax2.legend(loc='upper left')
            ax2.grid(True)

            # Formatting the date for the second subplot
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=45))
            ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

            ax2.set_xlabel('Date')
            ax2.set_ylabel('Value')

            # Adjusting layout
            plt.tight_layout()

            # Show the plot
            plt.show()


            C = df["Close"].dropna().tolist()
            M = df["MACD"].dropna().tolist()
            S = df["Signal"].dropna().tolist()
            H = df["Histogram"].dropna().tolist()


            from statsmodels.tsa.arima.model import ARIMA
            from pmdarima import auto_arima

            split_percentage = 0.80  # % for training
            split_index = int(len(H) * split_percentage)

            H_train = M[:split_index]
            H_test = M[split_index:]
            print(len(H_train), len(H_test))

            split_percentage = 0.80  # % for training
            split_index = int(len(C) * split_percentage)

            C_train = M[:split_index]
            C_test = M[split_index:]
            print(len(C_train), len(C_test))

            split_percentage = 0.80  # % for training
            split_index = int(len(M) * split_percentage)

            M_train = M[:split_index]
            M_test = M[split_index:]
            print(len(M_train), len(M_test))

            split_percentage = 0.80  # % for training
            split_index = int(len(S) * split_percentage)

            S_train = S[:split_index]
            S_test = S[split_index:]
            print(len(S_train), len(S_test))

            stepwise_fit = auto_arima(H,trace=True,suppress_warnings=True)
            stepwise_fit

            def extract_best_arima_order(stepwise_fit):
                # Search for the line starting with "Best model:"
                for line in stepwise_fit.split('
'):
                    if line.startswith("Best model:"):
                        # Extract numbers within parentheses
                        order = tuple(map(int, line.split('ARIMA')[1].split('(')[1].split(')')[0].split(',')))
                        return order
                return None
            arima_order = stepwise_fit.order
            arima_order
            hp, hd, hq = arima_order
            hp, hd, hq

            stepwise_fit = auto_arima(C,trace=True,suppress_warnings=True)
            stepwise_fit

            def extract_best_arima_order(stepwise_fit):
                # Search for the line starting with "Best model:"
                for line in stepwise_fit.split('
'):
                    if line.startswith("Best model:"):
                        # Extract numbers within parentheses
                        order = tuple(map(int, line.split('ARIMA')[1].split('(')[1].split(')')[0].split(',')))
                        return order
                return None
            arima_order = stepwise_fit.order
            arima_order
            cp, cd, cq = arima_order
            cp, cd, cq

            stepwise_fit = auto_arima(M,trace=True,suppress_warnings=True)
            stepwise_fit

            def extract_best_arima_order(stepwise_fit):
                # Search for the line starting with "Best model:"
                for line in stepwise_fit.split('
'):
                    if line.startswith("Best model:"):
                        # Extract numbers within parentheses
                        order = tuple(map(int, line.split('ARIMA')[1].split('(')[1].split(')')[0].split(',')))
                        return order
                return None
            arima_order = stepwise_fit.order
            arima_order
            mp, md, mq = arima_order
            mp, md, mq

            stepwise_fit = auto_arima(S,trace=True,suppress_warnings=True)
            stepwise_fit

            def extract_best_arima_order(stepwise_fit):
                # Search for the line starting with "Best model:"
                for line in stepwise_fit.split('
'):
                    if line.startswith("Best model:"):
                        # Extract numbers within parentheses
                        order = tuple(map(int, line.split('ARIMA')[1].split('(')[1].split(')')[0].split(',')))
                        return order
                return None
            arima_order = stepwise_fit.order
            arima_order
            sp, sd, sq = arima_order
            sp, sd, sq


            import statsmodels.api as sm

            # Define the order for ARIMA and seasonal order for seasonal component
            # For demonstration, I'll use a seasonal order of (1, 1, 1, 22), assuming 22 trading days in one month seasonality
            # You might need to adjust these parameters based on your specific dataset and domain knowledge

            arima_order = arima_order
            seasonal_order = (hp, hd, hq, SN)  

            model = sm.tsa.statespace.SARIMAX(H, order=arima_order, seasonal_order=seasonal_order)
            model = model.fit()

            start=len(H_train)
            end=len(H_train)+len(C_test)-1
            Hpred = model.predict(start=start,end=end)
            Hpred

            Hpred_future = model.predict(start=end,end=end+DD)
            Hpred_future


            import statsmodels.api as sm

            # Define the order for ARIMA and seasonal order for seasonal component
            # For demonstration, I'll use a seasonal order of (1, 1, 1, 22), assuming 22 trading days in one month seasonality
            # You might need to adjust these parameters based on your specific dataset and domain knowledge

            arima_order = arima_order
            seasonal_order = (cp, cd, cq, SN)  

            model = sm.tsa.statespace.SARIMAX(C, order=arima_order, seasonal_order=seasonal_order)
            model = model.fit()

            start=len(C_train)
            end=len(C_train)+len(C_test)-1
            Cpred = model.predict(start=start,end=end)
            Cpred

            Cpred_future = model.predict(start=end,end=end+DD)
            Cpred_future


            import statsmodels.api as sm

            # Define the order for ARIMA and seasonal order for seasonal component
            # For demonstration, I'll use a seasonal order of (1, 1, 1, 22), assuming 22 trading days in one month seasonality
            # You might need to adjust these parameters based on your specific dataset and domain knowledge

            arima_order = arima_order
            seasonal_order = (mp, md, mq, SN)  

            model = sm.tsa.statespace.SARIMAX(M, order=arima_order, seasonal_order=seasonal_order)
            model = model.fit()

            start=len(M_train)
            end=len(M_train)+len(M_test)-1
            Mpred = model.predict(start=start,end=end)
            Mpred

            Mpred_future = model.predict(start=end,end=end+DD)
            Mpred_future



            import statsmodels.api as sm

            # Define the order for ARIMA and seasonal order for seasonal component
            # For demonstration, I'll use a seasonal order of (1, 1, 1, 22), assuming 22 trading days in one month seasonality
            # You might need to adjust these parameters based on your specific dataset and domain knowledge

            arima_order = arima_order
            seasonal_order = (sp, sd, sq, SN)  

            model = sm.tsa.statespace.SARIMAX(S, order=arima_order, seasonal_order=seasonal_order)
            model = model.fit()

            start=len(S_train)
            end=len(S_train)+len(S_test)-1
            Spred = model.predict(start=start,end=end)
            Spred

            Spred_future = model.predict(start=end,end=end+DD)
            Spred_future




            # Assuming df.index is already set to datetime
            last_date = df.index[-2]  # Get the second to last date from the index
            start_date = last_date + pd.tseries.offsets.BDay(1)  # Calculate the start date as one business day after the last date
            dates = [start_date + pd.Timedelta(days=idx) for idx in range(43)]  # Generate a list of dates starting from 'start_date'


            # Filter out the weekend dates from the list
            market_dates = [date for date in dates if date.weekday() < 5]

            Date = pd.Series(market_dates )
            Date

            df3 = pd.DataFrame({'Date':Date,'Cpred_future': Cpred_future, 'Mpred_future': Mpred_future, 'Spred_future': Spred_future, 'Hpred_future': Hpred_future})
            df3

            import matplotlib.pyplot as plt
            import pandas as pd
            today = datetime.datetime.now().strftime("%Y-%m-%d")

            # Assuming df3 and df are already defined and Ticker is defined

            # Convert 'Date' to datetime if it's not already
            df3['Date'] = pd.to_datetime(df3['Date'])

            # Set the 'Date' column as the index of the DataFrame
            df3.set_index('Date', inplace=True)

            # Create a figure and a set of subplots
            fig, axs = plt.subplots(2, 1, figsize=(20, 12))  # Adjusts the figure size for two subplots

            # Plotting M and S predictions on the first subplot
            axs[0].plot(df3.index, df3['Mpred_future'], label='M predictions', marker='o', color='blue')
            axs[0].plot(df3.index, df3['Spred_future'], label='S predictions', marker='x', color='red')
            axs[0].set_title(f'Zoomed Forecast MACD of {Ticker}')
            axs[0].set_xlabel('Date')
            axs[0].set_ylabel('Values')
            axs[0].legend()
            axs[0].grid(True)
            axs[0].tick_params(axis='x', rotation=45)

            # Plotting Close Future predictions on the second subplot
            axs[1].plot(df3.index, df3['Cpred_future'], label='Close Future', marker='o', color='blue')
            axs[1].set_title(f'Zoomed Forecast Closing of {Ticker} Start Date {start_date1}')
            axs[1].set_xlabel('Date')
            axs[1].set_ylabel('Values')
            axs[1].legend()
            axs[1].grid(True)
            axs[1].tick_params(axis='x', rotation=45)

            plt.tight_layout()  # Adjusts the subplot params so that subplots are nicely fit in the figure

            # Construct the filename with today's date and ticker symbol
            filename = f"{today}_{Ticker}_Forecast.png"

            # Construct the path
            path = f"/Users/rajghotra/Library/CloudStorage/GoogleDrive-enterpriseghotra@gmail.com/My Drive/FileDrop/{Ticker}/{filename}"

            # Extract the directory path
            dir_path = os.path.dirname(path)

            # Check if the directory exists, if not create it
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Directory {dir_path} created.")
            else:
                print(f"Directory {dir_path} already exists.")

            plt.savefig(path)

            # Print the path for confirmation
            print(f"File saved as: {path}")
            plt.show()

            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import pandas as pd
            import numpy as np
            today = datetime.datetime.now().strftime("%Y-%m-%d")

            # Assuming 'df' and 'df3' are your DataFrames, and their indexes are of datetime type.
            # Also assuming 'df' already has 'MACD', 'Signal', 'Close' columns calculated.
            # 'df3' should contain your future predictions 'Mpred_future', 'Spred_future', 'Cpred_future', 'Hpred_future'.

            # Sorting the DataFrame by the index (date) to ensure the data is in chronological order
            df = df.sort_index()

            # Calculate the Histogram if not already done
            df['Histogram'] = df['MACD'] - df['Signal']

            # Creating a figure and a grid of subplots
            fig, axs = plt.subplots(5, 1, figsize=(14.875, 19.25), dpi=300)
            fig.suptitle(f"{Ticker}-Data Used for Forecasting {start_date1} to {today} for {DD} Days Forecast", fontsize=25, y=.99)
            # Plotting the Close price on axs[0]
            axs[4].plot(df.index, df['Close'], label='Close', color='Black')
            axs[4].set_title('Close Price')
            axs[4].legend(loc='upper left')
            axs[4].grid(True)

            # Formatting the date to ensure that the date does not overlap
            axs[4].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=45))
            axs[4].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[0].xaxis.get_major_locator()))

            # Plotting MACD and Signal Line on axs[1]
            axs[1].plot(df.index, df['MACD'], label='MACD', color='blue', linewidth=1.5)
            axs[1].plot(df.index, df['Signal'], label='Signal Line', color='red', linewidth=1.5)

            # Plotting the Histogram as bar plot on axs[1]
            axs[1].bar(df.index, df['Histogram'], label='Histogram', color='grey', alpha=0.3)

            axs[1].set_title('MACD, Signal Line, and Histogram')
            axs[1].legend(loc='upper left')
            axs[1].grid(True)

            # Formatting the date for the second subplot
            axs[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=45))
            axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))

            # Forecast plots
            # MACD and Signal future predictions
            axs[2].plot(df.index, df['MACD'], label='MACD', color='blue')
            axs[2].plot(df.index, df['Signal'], label='Signal', color='red')
            axs[2].plot(df3.index[-1000:], df3['Mpred_future'][-1000:], label='MACD Future', linestyle='--', color='blue')
            axs[2].plot(df3.index[-1000:], df3['Spred_future'][-1000:], label='Signal Future', linestyle='--', color='red')
            axs[2].set_title('Forecast MACD and Signal Line')
            axs[2].legend()

            # Closing price and future prediction
            axs[3].plot(df.index, df['Close'], label='Closed', color='Black')
            axs[3].plot(df3.index[-1000:], df3['Cpred_future'][-1000:], label='Closing Future', linestyle='--', color='Blue')
            axs[3].set_title('Forecast Closing Price')
            axs[3].legend()

            # Histogram and future prediction
            width = 0.22
            axs[0].bar(df.index, df['Histogram'], width, label='Histogram', color='blue')
            axs[0].bar(df3.index[-1000:], df3['Hpred_future'][-1000:], width, label='Histogram Future', color='green')
            axs[0].set_title('Forecast Histogram')
            axs[0].legend()

            # General settings for all subplots
            for ax in axs:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
                ax.grid(True)
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')

            plt.tight_layout(pad=1)

            # Construct the filename with today's date and ticker symbol
            filename = f"{today}_{Ticker}.png"

            # Construct the path
            path = f"/Users/rajghotra/Library/CloudStorage/GoogleDrive-enterpriseghotra@gmail.com/My Drive/FileDrop/{Ticker}/{filename}"

            # Extract the directory path
            dir_path = os.path.dirname(path)

            # Check if the directory exists, if not create it
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"Directory {dir_path} created.")
            else:
                print(f"Directory {dir_path} already exists.")

            plt.savefig(path)

            # Print the path for confirmation
            print(f"File saved as: {path}")







