
from unicorn_binance_websocket_api.unicorn_binance_websocket_api_manager import BinanceWebSocketApiManager
import pdb
from unicorn_fy.unicorn_fy import UnicornFy
import csv
import ccxt
exchange = ccxt.binance({'apiKey': "H9XUAlGqmqy9JSoHgp3BmBlv02Z1zYxeMA4aU6l66wtGBVs8Ki6789sRgErpf1wC", 'secret': "e9ueiRyLpDV96l3RMaircqvVvMOT0ygGyZmqEUnyf3IXjBulAp24MMQviygoagiZ"})
exchange.load_markets ()
symbols = exchange.symbols
btc_symbols = []
for symbol in symbols:
    if '/BTC' in symbol:
        btc_symbols.append(symbol.split('/')[0]+symbol.split('/')[1])

binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.com")
binance_websocket_api_manager.create_stream(['ticker'], btc_symbols )

while True:
    oldest_stream_data_from_stream_buffer = binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
    if oldest_stream_data_from_stream_buffer:
        unicorn_fied_stream_data = UnicornFy.binance_com_websocket(oldest_stream_data_from_stream_buffer)
        symbol = unicorn_fied_stream_data['data'][0]['symbol']
        fields=[symbol,
                unicorn_fied_stream_data['data'][0]['event_time'],
                unicorn_fied_stream_data['data'][0]['price_change'],
                unicorn_fied_stream_data['data'][0]['price_change_percent'],
                unicorn_fied_stream_data['data'][0]['last_price'],
                unicorn_fied_stream_data['data'][0]['best_bid_price'],
                unicorn_fied_stream_data['data'][0]['best_ask_price'],
                unicorn_fied_stream_data['data'][0]['total_traded_base_asset_volume'],
                unicorn_fied_stream_data['data'][0]['total_traded_quote_asset_volume']]
        with open(symbol + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

