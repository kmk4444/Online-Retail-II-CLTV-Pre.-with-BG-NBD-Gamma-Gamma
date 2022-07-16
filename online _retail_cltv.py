#Dataset History
#Online Retail II dataset contains all the transactions occurring in a UK-based, online retail between 01/12/2009 and 09/12/2011.

#Variable Information
#InvoiceNo: Invoice number. A 6-digit number uniquely assigned to each transaction. If this code starts with the letter 'C', it indicates a cancellation.
#StockCode: Product (item) code. A 5-digit number uniquely assigned to each distinct product.
#Description: Product (item) name.
#Quantity: The quantities of each product (item) per transaction.
#InvoiceDate: The day and time when a transaction was generated.
#UnitPrice: Product price per unit in sterlin.
#CustomerID: A 5-digit number uniquely assigned to each customer.
#Country: The name of the country where a customer resides.

#Business Problem
#UK based retail company wants to segment its customers and wants to determine its marketing strategies according to these segments to increase the company's revenue.
# For this purpose, they want to do behavioural segmentation with respect to customers' purchase behaviours and preferences.

###################################### TASK 1 ################################

#Step 1: estimate cltv prediction for six months by the means of using BG-NBD and Gamma-Gamma Methods

# !pip install lifetimes

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_excel("WEEK_3/Customer_life_Time_Value_Prediction/online_retail_II.xlsx", sheet_name= "Year 2010-2011")
df = df_.copy()
df.head()

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na = False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

# we can eliminate outlier for just integer variable

def outlier_threshold(dataframe, variable):
    # quantile provide to sort from smallest to largest;after that, we select quarter. (0.25- 0.75)
    # but we use (0.01-0.99) because this dataset do not have a large number frequency. We prefer to these ranges.
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable): # aykırı değer baskılama
    low_limit, up_limit = outlier_threshold(dataframe, variable) # threshold func. ile üst ve alt sınırlar belli olacaktır.
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # our dataset do not have negative value,therefore, we hava used.
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# we can eliminate outlier for just integer variables
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")


df["TotalPrice"] = df["Quantity"] * df["Price"]




###################################################################
# 1.Preparation of Lifetime Data Structure
###################################################################

today_date = dt.datetime(2011, 12 ,11)

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                                        lambda x: (today_date - x.min()).days],
                                        "Invoice": lambda x: x.nunique(),
                                        "TotalPrice" : lambda x: x.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0) # we delete up line

cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] # average benefit
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]
cltv_df["recency"] = cltv_df["recency"] / 7 # Week
cltv_df["T"] = cltv_df["T"] / 7 # Week
cltv_df.head()

###################################################################
# Establishment of BG-NBD Model
###################################################################

bgf = BetaGeoFitter(penalizer_coef=0.001) # ceza katsayısı

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# WHO ARE THE 10 CUSTOMERS WE EXPECT TO BUY THE MOST IN A WEEK?
bgf.conditional_expected_number_of_purchases_up_to_time(1, # week
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purch_6_months"] = bgf.predict(24, # haftalık cinsinden oluşturduğumuz için 1 haftalık tahmin et
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])

# our model has superb prediction in some points;nevertheless, the other points is bad.
plot_period_transactions(bgf)
plt.show()

###################################################################
# Establishing the Gamma - Gamma Model
###################################################################

ggf = GammaGammaFitter(penalizer_coef = 0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"])

####################################################################
#Calculation of CLTV with BG-NBD and GG Model
###################################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=6, # month
                                   freq = "W", # T
                                   discount_rate=0.01) # average discount rate

cltv.head()

cltv = cltv.reset_index() # customer id converts index into real value.

# in order that we analyze better, we combine cltv and cltv_df dataframes.
cltv_final = cltv_df.merge(cltv, on="Customer ID", how= "left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

###################################################################
# Creating the Customer Segment
###################################################################
cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending = False).head(10)


cltv_final.groupby("segment").agg({"count","mean","sum"})

#graph 1

cltv_final.groupby('segment').agg('expected_average_profit').mean().plot(kind='bar', colormap='copper_r');

plt.ylabel("profit");

#graph 2

cltv_final.groupby('segment').agg('expected_purch_6_months').mean().plot(kind='bar', colormap='copper_r');

plt.ylabel("expected purchase");

#graph 3

cltv_final.groupby('segment').agg('clv').mean().plot(kind='bar', colormap='copper_r');

plt.ylabel("clv");
