from dateutil import parser
from configparser import ConfigParser
import pandas as pd
import pdb
#import ccxt
import time
import numpy as np
from pycoingecko import CoinGeckoAPI


def collectdata():
	cg = CoinGeckoAPI()
	pdb.set_trace()
	data = cg.get_exchanges_tickers_by_id("binance")

if __name__ == '__main__':
	collectdata()