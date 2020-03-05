import platform
import numpy as np
import matplotlib
import pdb
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import pyplot as plt
if platform.platform() == "Darwin-18.7.0-x86_64-i386-64bit":
    matplotlib.use("macOSX")


def plot_whole(df,end,pattern,window):
    lookforward = 0
    fragment = df.iloc[end-window:end+1+lookforward,:]
    fragment = fragment.reset_index()
    fragment = fragment.fillna(0)
    symbol = df.iloc[-1].symbol
    plt.clf()
    fig,axes = plt.subplots(nrows=2, ncols=1)
    fragment.total_traded_quote_asset_volume.plot(ax=axes[0] , color="blue", style='.-')
    fragment.qav_sma50.plot(ax=axes[0], color="red")
    fragment.qav_sma100.plot(ax=axes[0], color="orange")
    fragment.qav_sma200.plot(ax=axes[0], color="brown")
    fragment.last_price.plot(ax=axes[1], style='.-')
    fragment.last_sma100.plot(ax=axes[1], color="yellow")
    fragment.last_sma200.plot(ax=axes[1], color="purple")
    fragment.last_sma600.plot(ax=axes[1], color="black")
    plt.title(symbol)
    axes[0].plot(window,fragment.iloc[-1-lookforward].total_traded_quote_asset_volume,'g*',color="red")
    axes[1].plot(window,fragment.iloc[-1-lookforward].last_price,'g*',color="red")
    #plt.show()
    main_path = "/Users/apple/Desktop/dev/projectlife/hf/patterns/"
    path = main_path + pattern + "/"+symbol+"-"+str(end)+"-"+str(window)+".png"
    plt.savefig(path)

with open('/Users/apple/Desktop/dev/projectlife/hf/patterns/spike_patterns.json') as json_file:
    patterns = json.load(json_file)
    for pattern in patterns:
        data_base = pd.read_csv("/Users/apple/Desktop/dev/projectlife/data/" +  pattern['data'] +"/"+pattern['symbol']+".csv")
        df = pd.DataFrame(data_base)
        df.columns = ['symbol','date','price_change','price_change_percent','last_price','best_bid_price','best_ask_price','total_traded_base_asset_volume','total_traded_quote_asset_volume']
        df['qav_sma50'] = df.total_traded_quote_asset_volume.rolling(50).mean()
        df['qav_sma100'] = df.total_traded_quote_asset_volume.rolling(100).mean()
        df['qav_sma200'] = df.total_traded_quote_asset_volume.rolling(200).mean()
        df['last_sma100'] = df.last_price.rolling(100).mean()
        df['last_sma200'] = df.last_price.rolling(200).mean()
        df['last_sma600'] = df.last_price.rolling(600).mean()

        plot_whole(df,pattern["end"], pattern['type'],1000)
        plot_whole(df,pattern["end"],pattern['type'],2000)
        plot_whole(df,pattern["end"],pattern['type'],3000)
        plot_whole(df,pattern["end"],pattern['type'],4000)

