import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyportfolioopt import weights_max_sharpe, plot_efficient_frontier
from fundamental_score import get_score
from ytd_performance import get_ytd_performance
import random
import requests
import json
#from plotly import graph_objects

st.markdown("""# AI-Powered Stock Portfolio Manager 
Invest with the odds in your favor. 
Get unique insights, boost your 
portfolios, and make smart data-driven investment decisions.""")

# Adding a text in the sidebar
# st.sidebar.text('Your Text Here')
# # Add a radio button
# st.sidebar.radio('label', options=[])

#st.markdown("""## Stocks Ranked by AI
#US-listed stocks are ranked according to the AI Score, which rates the probability of beating the market in the next 3 months.

#""")
st.header("Top 10 AI Recommended Stocks for the Next 1 Month")
st.markdown("""
US-listed stocks are ranked according to the Piotroski Score, which takes into account of 3 criterias i.e. Profitability, Operating Efficiency & Leverage/Liquidity/Source of Funds
""")
tick_data=pd.read_csv('https://raw.githubusercontent.com/kychee2013/Stock_Portfolio_Prediction/main/Streamlit/sp500_companies.csv')
result_df = pd.read_csv('https://raw.githubusercontent.com/kychee2013/Stock_Portfolio_Prediction/main/Streamlit/result.csv')
techscore_df = pd.read_csv("https://raw.githubusercontent.com/kychee2013/Stock_Portfolio_Prediction/main/Streamlit/techscore.csv")
allscores_df = pd.read_csv("https://raw.githubusercontent.com/kychee2013/Stock_Portfolio_Prediction/main/Streamlit/scores.csv")
ytd_df = pd.read_csv("https://raw.githubusercontent.com/kychee2013/Stock_Portfolio_Prediction/main/Streamlit/top10_ytd.csv")
# techscore_df = pd.read_csv("scores.csv")
tickers = tick_data["Ticker"].tolist()
company_name = tick_data["Company Name"].tolist()
sector = tick_data["Sector"].tolist()

data = {
    'Company': tickers,
    'Company Name': company_name,
    'Sector':sector
}

# Create a DataFrame from the data

with st.spinner("Loading, please wait..."):
    
    df = pd.DataFrame(data).set_index('Company')
    df = df[~df.index.duplicated(keep='first')]
    techscore_df = techscore_df[~techscore_df.index.duplicated(keep='first')]
    # df = pd.concat([df,techscore_df.set_index("Ticker")["Score"]]).rename(columns={"Score":"Technical Score"})
    # df = pd.concat([df,techscore_df.set_index("Tickers")]).rename(columns={"Score":"Technical Score"})
    df = pd.concat([df, techscore_df.set_index("Ticker")["Score"]], axis=1, join="outer")
    df.rename(columns={"Score": "Technical Score"}, inplace=True)


    #read from csv
    df = pd.concat([df, result_df.set_index("tickers")], axis=1)
    flag = df.drop(['Company Name', 'Sector'], axis=1).T.sum().sort_values(ascending=False).index[:10]
    df = df.loc[flag]
    df = pd.concat([df, ytd_df.set_index("Ticker")["YTD Performance"]], axis=1)
    df.rename(columns={"YTD Performance":"YTD Performance (%)"}, inplace=True)
    df["YTD Performance (%)"] = df["YTD Performance (%)"].astype('int32')
    #df["YTD Performance (%)"].astype(int)
    # df["YTD Performance"] = [get_ytd_performance(ticker) for ticker in flag]
    show_df = df.rename(columns={"Sum": 'Fundamental Score'}).reset_index().rename(columns={"index": 'Company'})
    show_df["Technical Score"] = show_df["Technical Score"].map(lambda x: int(x))
    show_df["Fundamental Score"] = show_df["Fundamental Score"].map(lambda x: int(x))
    show_df["Rank"] = [_+1 for _ in range(df.shape[0])]
    # Display the ranked stock table
    # st.dataframe(show_df.set_index("Rank"),width=1000)
    
    st.dataframe(show_df.set_index("Rank").style.set_table_styles([{
        'selector': 'td',
        'props': [('max-width', '200px')]
    }]))
    

# My stock portfolio optimizer
st.header("My Stock Portfolio Optimizer")

user_tickers = st.multiselect('Select all stock tickers to be included in portfolio separated by commas',
                              tickers)
if st.button('Generate'):
    with st.spinner('Building, please wait'):

        company_name = tick_data[tick_data["Ticker"].isin(user_tickers)]["Company Name"].tolist()
        sector = tick_data[tick_data["Ticker"].isin(user_tickers)]["Sector"].tolist()
        
        data = {
            'Company': user_tickers,
            'Company Name': company_name,
            'Sector':sector
        }

        df = pd.DataFrame(data).set_index('Company')
        
        df = pd.concat([df, allscores_df.set_index("Ticker")["Score"]], axis=1, join="inner")
        df.rename(columns={"Score": "Technical Score"}, inplace=True)
        df = pd.concat([df, result_df.set_index("tickers")], axis=1, join='inner')
        #flag = df.drop(['Company Name', 'Sector'], axis=1).T.sum().sort_values(ascending=False).index[:10]
        #df = df.loc[flag]
        df = pd.concat([df, ytd_df.set_index("Ticker")["YTD Performance"]], axis=1, join='inner')
        df.rename(columns={"YTD Performance":"YTD Performance (%)"}, inplace=True)
        #df["YTD Performance (%)"] = df["YTD Performance (%)"].astype('int32')
        # df["YTD Performance"] = [get_ytd_performance(ticker) for ticker in flag]
        #show_df = df.rename(columns={"Sum": 'Fundamental Score'}).reset_index().rename(columns={"index": 'Company'})
        show_df["Technical Score"] = show_df["Technical Score"].map(lambda x: int(x))
        #show_df["Fundamental Score"] = show_df["Fundamental Score"].map(lambda x: int(x))
        #show_df["Rank"] = [_+1 for _ in range(df.shape[0])]

        #filtered_df = show_df[show_df['Company'].isin(user_tickers)]
        filtered_df = show_df

        # Reset the index to update the ranking
        filtered_df = filtered_df.reset_index(drop=True)
        st.dataframe(filtered_df)
        #st.dataframe(filtered_df.set_index("Rank"))
        
       

        predicted_stocks=requests.get("http://34.125.120.216:8000/predict?ticker={}".format(",".join(user_tickers)))
        
        # Convert predicted 1 month Close price to DataFrame
        predicted_1mo = {}
        predicted_stocks=json.loads(predicted_stocks.text)

        for stock in predicted_stocks:
            predicted_1mo[stock] = predicted_stocks[stock]
        mu, sigma, sharpe, a = weights_max_sharpe(pd.DataFrame.from_dict(predicted_1mo), 21)

        st.header("Portfolio's Predicted Performance")
        
        st.subheader("Expected annual return: {:.1f}%".format(100 * mu))
        st.markdown("Predicted % return (annualized) that can be expected from this portfolio")
        
        st.subheader("Annual volatility: {:.1f}%".format(100 * sigma))
        st.markdown("Measure of the dispersion of returns for a stock portfolio.\n"
        "Measured from the standard deviation between returns from that same stock portfolio.\n"
        "The higher the volatility, the riskier the security.")

        #sharpe_dict = {
        #    range(0,1): 'Sub-optimal',
        #    range(1,2): 'Good',
        #    range(2,3): 'Very good',
        #    range(3,1000): 'Excellent'
        #}
        
        st.subheader("Sharpe Ratio: {:.2f}".format(sharpe))
        #st.subheader("({score})".format(score = sharpe_dict[int(sharpe)]))
        st.markdown("Difference between the risk-free return and the return of a portfolio divided by the portfolio’s standard deviation.\n"
        "It is often used to carry out the performance of a particular share against the risk.\n"
        "Usually, any Sharpe ratio greater than 1.0 is considered acceptable to good by investors.\n"
        "- A ratio higher than 2.0 is rated as very good.\n"
        "- A ratio of 3.0 or higher is considered excellent.\n"
        "- A ratio under 1.0 is considered sub-optimal.")

        st.header("Portfolio Weightage (%)")
        st.markdown("""Note: Portfolio weightage shown below is optimized for Sharpe Ratio i.e. portfolio with the maximum return-risk ratio""")
        
        sizes = [a[_] for _ in a.keys() if a[_] != 0]
        labels = ["{}\n{}%".format(_, round(a[_]*100)) for _ in a.keys() if a[_] != 0]
        fig1, axes1 = plt.subplots(figsize=(10, 10))

        wedges, texts = axes1.pie(sizes, labels=labels, startangle=90)
        for label, text in zip(labels, texts):
            text.set(size=15, color='black')
        axes1.set_title('Portfolio Weightage (%)', fontsize = 20)
        st.pyplot(fig1)

        #plot_efficient_frontier(pd.DataFrame.from_dict(predicted_1mo), 21)

        st.header("Predicted 1 Month Stock Close Price USD ($)")
        
        # Line chart for predicted stock prices
        fig2, axes2 = plt.subplots(figsize=(10, 10))
        for stock in predicted_stocks:
            axes2.plot(range(1, 22), predicted_1mo[stock], label = stock)

        axes2.spines['top'].set_visible(False)
        axes2.spines['right'].set_visible(False)
        axes2.tick_params(axis='both', which='major', labelsize=10)
        axes2.set_xlabel('Days from Today', fontsize = 15)
        axes2.set_ylabel('Predicted Stock Close Price USD ($)', fontsize = 15)
        axes2.set_title('Predicted 1 Month Stock Prices USD ($)', fontsize = 20)
        axes2.legend()
        st.pyplot(fig2)

        
