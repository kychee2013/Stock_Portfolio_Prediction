import pandas as pd


def info_filter(df, stats, indx, lookback):
    df_new = df.loc[stats, df.columns[:3]]
    df_new.rename(dict(zip(stats, indx)), inplace=True)
    df_new.loc["OtherLTDebt", :] = df_new.loc["TotLTLiab", :] - df_new.loc["LTDebt", :]
    return df_new


def piotroski_f(df_dict):
    f_score = {}
    for ticker in df_dict:
        columns = df_dict[ticker].columns
        ROA_FS = int(df_dict[ticker].loc["NetIncome", columns[0]] / ((df_dict[ticker].loc["TotAssets", columns[0]] +
                                                                      df_dict[ticker].loc[
                                                                          "TotAssets", columns[1]]) / 2) > 0)
        CFO_FS = int(df_dict[ticker].loc["CashFlowOps", columns[0]] > 0)
        ROA_D_FS = int(
            (df_dict[ticker].loc["NetIncome", columns[0]] / ((df_dict[ticker].loc["TotAssets", columns[0]] +
                                                              df_dict[ticker].loc[
                                                                  "TotAssets", columns[1]]) / 2)) > (
                    df_dict[ticker].loc["NetIncome", columns[1]] / ((df_dict[ticker].loc[
                                                                         "TotAssets", columns[1]] +
                                                                     df_dict[ticker].loc[
                                                                         "TotAssets", columns[2]]) / 2)))
        CFO_ROA_FS = int(
            df_dict[ticker].loc["CashFlowOps", columns[0]] / df_dict[ticker].loc["TotAssets", columns[0]] >
            df_dict[ticker].loc["NetIncome", columns[0]] / ((df_dict[ticker].loc["TotAssets", columns[0]] +
                                                             df_dict[ticker].loc[
                                                                 "TotAssets", columns[1]]) / 2))
        LTD_FS = int(
            (df_dict[ticker].loc["LTDebt", columns[0]] + df_dict[ticker].loc["OtherLTDebt", columns[0]]) < (
                    df_dict[ticker].loc["LTDebt", columns[1]] + df_dict[ticker].loc["OtherLTDebt", columns[1]]))
        CR_FS = int(
            (df_dict[ticker].loc["CurrAssets", columns[0]] / df_dict[ticker].loc["CurrLiab", columns[0]]) > (
                    df_dict[ticker].loc["CurrAssets", columns[1]] / df_dict[ticker].loc["CurrLiab", columns[1]]))
        DILUTION_FS = int(
            df_dict[ticker].loc["CommStock", columns[0]] <= df_dict[ticker].loc["CommStock", columns[1]])
        GM_FS = int(
            (df_dict[ticker].loc["GrossProfit", columns[0]] / df_dict[ticker].loc["TotRevenue", columns[0]]) > (
                    df_dict[ticker].loc["GrossProfit", columns[1]] / df_dict[ticker].loc["TotRevenue", columns[1]]))
        ATO_FS = int(
            (df_dict[ticker].loc["TotRevenue", columns[0]] / ((df_dict[ticker].loc["TotAssets", columns[0]] +
                                                               df_dict[ticker].loc[
                                                                   "TotAssets", columns[1]]) / 2)) > (
                    df_dict[ticker].loc["TotRevenue", columns[1]] / ((df_dict[ticker].loc[
                                                                          "TotAssets", columns[1]] +
                                                                      df_dict[ticker].loc[
                                                                          "TotAssets", columns[2]]) / 2)))
        f_score[ticker] = [ROA_FS, CFO_FS, ROA_D_FS, CFO_ROA_FS, LTD_FS, CR_FS, DILUTION_FS, GM_FS, ATO_FS]
    f_score_df = pd.DataFrame(f_score,
                              index=["PosROA", "PosCFO", "ROAChange", "Accruals", "Leverage", "Liquidity",
                                     "Dilution",
                                     "GM", "ATO"])
    return f_score_df


def get_score(tickers):
    financial_dir = {}

    for ticker in tickers:
        try:
            print("scraping financial statement data for ", ticker)
            url = "https://stockrow.com/api/companies/{}/financials.xlsx?dimension=A&section=Balance%20Sheet&sort=desc".format(ticker)
            df1 = pd.read_excel(url)
            url = ("https://stockrow.com/api/companies/{}/financials.xlsx?dimension=A&section=Income%20Statement&sort=desc").format(ticker)
            df2 = pd.read_excel(url)
            url = "https://stockrow.com/api/companies/{}/financials.xlsx?dimension=A&section=Cash%20Flow&sort=desc".format(ticker)
            df3 = pd.read_excel(url)
            df = pd.concat([df1, df2, df3])
            columns = df.columns.values
            for i in range(len(columns)):
                if columns[i] == "Unnamed: 0":
                    columns[i] = "heading"
                else:
                    columns[i] = columns[i].strftime("%Y-%m-%d")
            df.columns = columns
            df.set_index("heading", inplace=True)
            financial_dir[ticker] = df
        except Exception as e:
            print(ticker, ":", e)

    stats = ["Net Income Common",
             "Total Assets",
             "Operating Cash Flow",
             "Long Term Debt (Total)",
             "Total non-current liabilities",
             "Total current assets",
             "Total current liabilities",
             "Common Equity (Total)",
             "Revenue",
             "Gross Profit"]

    indx = ["NetIncome", "TotAssets", "CashFlowOps", "LTDebt", "TotLTLiab",
            "CurrAssets", "CurrLiab", "CommStock", "TotRevenue", "GrossProfit"]

    transformed_df = {}
    for ticker in financial_dir:
        transformed_df[ticker] = info_filter(financial_dir[ticker], stats, indx, 3)
        
    print(transformed_df)
    
    f_score_df = piotroski_f(transformed_df)
    
    print(f_score_df)
    
    return f_score_df.sum().sort_values(ascending=False)
