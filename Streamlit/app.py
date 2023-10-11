import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pyportfolioopt import weights_max_sharpe, plot_efficient_frontier
from fundamental_score import get_score
import random
import requests
import json

st.markdown("""# AI-Powered Stock Picking 
Invest with the odds in your favor. 
Get unique insights, boost your 
portfolios, and make smart data-driven investment decisions.""")

# Adding a text in the sidebar
# st.sidebar.text('Your Text Here')
# # Add a radio button
# st.sidebar.radio('label', options=[])

st.markdown("""## Stocks Ranked by AI
US-listed stocks are ranked according to the AI Score, which rates the probability of beating the market in the next 3 months.

""")
tick_data=pd.read_csv('kychee2013/Stock_Portfolio_Prediction/Streamlit/sp500_companies.csv')
result_df = pd.read_csv('result.csv')
techscore_df = pd.read_csv("techscore.csv")
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
    df["YTD Performance"] = [get_ytd_performance(ticker) for ticker in flag]
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
        data = {
            'Company': user_tickers,
        }

        df = pd.DataFrame(data).set_index('Company')
        
        df["Technical Score"] = [random.randint(1, 11) for _ in range(df.shape[0])]
        #read from csv
        df = pd.concat([df, result_df.set_index("tickers").loc[user_tickers]], axis=1)
        flag = df.T.sum().sort_values(ascending=False).index
        df = df.loc[flag]
        # sort_inx = df.T.sum().sort_values(ascending=False).index
        df["YTD Performance"] = [get_ytd_performance(ticker) for ticker in flag]
        show_df = df.rename(columns={"Sum": 'Fundamental Score'}).reset_index().rename(columns={"index": 'Company'})
        
        show_df["Rank"] = [_+1 for _ in range(df.shape[0])]
        # Display the ranked stock table
        # st._legacy_dataframe(show_df.set_index("Rank"))
        st.dataframe(show_df.set_index("Rank").style.set_table_styles([{
            'selector': 'td',
            'props': [('max-width', '100px')]
        }]))
       

        predicted_stocks=requests.get("http://34.125.120.216:8000/predict?ticker={}".format(",".join(user_tickers)))
        
        # Convert predicted 1 month Close price to DataFrame
        predicted_1mo = {}
        predicted_stocks=json.loads(predicted_stocks.text)

        for stock in predicted_stocks:
            predicted_1mo[stock] = predicted_stocks[stock]
        mu, sigma, sharpe, a = weights_max_sharpe(pd.DataFrame.from_dict(predicted_1mo), 21)
        st.subheader("Expected annual return: {:.1f}%".format(100 * mu))
        st.subheader("Annual volatility: {:.1f}%".format(100 * sigma))
        st.subheader("Sharpe Ratio: {:.2f}".format(sharpe))

        sizes = [a[_] for _ in a.keys() if a[_] != 0]
        labels = ["{}\n{}%".format(_, round(a[_]*100)) for _ in a.keys() if a[_] != 0]
        fig, axes = plt.subplots(figsize=(10, 10))

        wedges, texts = axes.pie(sizes, labels=labels, startangle=90)
        for label, text in zip(labels, texts):
            text.set(size=15, color='black')
        st.pyplot(fig)

        plot_efficient_frontier(pd.DataFrame.from_dict(predicted_1mo), 21)
