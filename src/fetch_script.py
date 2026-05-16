from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']
START = date(2024, 3, 1)
DAYS = 7
OUT_DIR = Path('data/trades')

OUT_DIR.mkdir(parents=True, exist_ok=True)

for symbol in SYMBOLS:
    for i in range(DAYS):
        day = START + timedelta(days=i)
        out_path = OUT_DIR / f'{symbol}-{day}.parquet'

        if out_path.exists():
            print(f'Skipping {symbol} {day} (exists)')
            continue

        url = f'https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{day}.zip'
        print(f'Downloading {symbol} {day}...')

        try:
            df = pd.read_csv(
                url,
                compression='zip',
                names=['trade_id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']
            )
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.to_parquet(out_path)
            print(f'  → {len(df):,} trades')
        except Exception as e:
            print(f'  ✗ Failed: {e}')
