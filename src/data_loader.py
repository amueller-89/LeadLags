"""
Binance data loader for cryptocurrency OHLCV data.

Fetches historical data using ccxt and saves to parquet format.
"""

import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
import time


class BinanceDataLoader:
    """Fetches and manages Binance cryptocurrency data."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory to save downloaded data
        """
        self.exchange = ccxt.binance({
            'enableRateLimit': True,  # Respect Binance rate limits
            'options': {'defaultType': 'spot'}  # Use spot market
        })
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', etc.)
            start_date: Start date for data fetch
            end_date: End date for data fetch
            days_back: If start_date not provided, fetch this many days back

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Set date range
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=days_back)

        since = int(start_date.timestamp() * 1000)  # ccxt uses milliseconds
        end_ms = int(end_date.timestamp() * 1000)

        print(f"Fetching {symbol} {timeframe} data from {start_date} to {end_date}")

        all_candles = []
        current_since = since

        # Binance returns max 1000 candles per request
        while current_since < end_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Update since to last candle timestamp + 1
                current_since = candles[-1][0] + 1

                # Rate limiting - small delay between requests
                time.sleep(self.exchange.rateLimit / 1000)

                # Stop if we've reached the end date
                if candles[-1][0] >= end_ms:
                    break

            except ccxt.NetworkError as e:
                print(f"Network error: {e}, retrying...")
                time.sleep(5)
                continue
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {e}")
                raise

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Filter to exact date range
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        print(f"Fetched {len(df)} candles for {symbol}")

        return df

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1m',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days_back: int = 7,
        save: bool = True
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
            timeframe: Candle timeframe
            start_date: Start date for data fetch
            end_date: End date for data fetch
            days_back: If start_date not provided, fetch this many days back
            save: Whether to save data to parquet files

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    days_back=days_back
                )
                data[symbol] = df

                if save:
                    self.save_to_parquet(df, symbol, timeframe)

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue

        return data

    def save_to_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Path:
        """
        Save DataFrame to parquet file.

        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            timeframe: Timeframe of the data

        Returns:
            Path to saved file
        """
        # Clean symbol for filename (replace / with _)
        clean_symbol = symbol.replace('/', '_')

        # Create filename with date range
        start_date = df.index.min().strftime('%Y%m%d')
        end_date = df.index.max().strftime('%Y%m%d')
        filename = f"{clean_symbol}_{timeframe}_{start_date}_{end_date}.parquet"

        filepath = self.data_dir / filename
        df.to_parquet(filepath, compression='snappy')

        print(f"Saved to {filepath}")
        return filepath

    def load_from_parquet(self, filepath: str) -> pd.DataFrame:
        """
        Load DataFrame from parquet file.

        Args:
            filepath: Full path to the parquet file

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_parquet(filepath)

        # Ensure timestamp is the index if it exists as a column
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        return df

    def fetch_trades(
        self,
        symbol: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        hours_back: float = 1.0,
        save: bool = False
    ) -> pd.DataFrame:
        """
        Fetch tick/trade data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_ms: Start timestamp in milliseconds (UTC)
            end_ms: End timestamp in milliseconds (UTC)
            hours_back: If start_ms not provided, fetch this many hours back
            save: Whether to save to parquet

        Returns:
            DataFrame with columns: timestamp, price, amount, cost, side, id

        Note:
            Binance API typically only provides recent trades (last 24 hours).
            For historical data, download from https://data.binance.vision/
        """
        # Determine time range
        if end_ms is None:
            end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        if start_ms is None:
            start_ms = end_ms - int(hours_back * 3600 * 1000)

        print(f"Fetching trades for {symbol} from {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc)} to {datetime.fromtimestamp(end_ms/1000, tz=timezone.utc)}")

        all_trades = []
        current_since = start_ms
        batch_count = 0

        try:
            while current_since < end_ms:
                batch_count += 1

                # Fetch trades (limit=1000 is Binance max)
                trades = self.exchange.fetch_trades(
                    symbol,
                    since=current_since,
                    limit=1000
                )

                if not trades:
                    print("No more trades available")
                    break

                # Filter trades within our time range
                valid_trades = [t for t in trades if t['timestamp'] < end_ms]
                all_trades.extend(valid_trades)

                # Update since to last trade timestamp + 1ms
                if trades:
                    last_timestamp = trades[-1]['timestamp']
                    if last_timestamp >= current_since:
                        current_since = last_timestamp + 1
                    else:
                        break  # No progress, avoid infinite loop

                    print(f"Batch {batch_count}: Fetched {len(trades)} trades, total: {len(all_trades)}")
                else:
                    break

                # Safety limit to avoid excessive fetching
                if batch_count > 1000:
                    print(f"Warning: Reached safety limit of 1000 batches")
                    break

                # Small delay to respect rate limits
                time.sleep(0.1)

        except Exception as e:
            print(f"Error fetching trades: {e}")
            if all_trades:
                print(f"Returning {len(all_trades)} trades fetched before error")
            else:
                raise

        if not all_trades:
            raise ValueError(f"No trade data available for {symbol} in specified time range")

        # Convert to DataFrame
        df = pd.DataFrame(all_trades)

        # Extract relevant columns
        df = df[['timestamp', 'price', 'amount', 'cost', 'side', 'id']]

        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)

        # Sort by timestamp
        df.sort_index(inplace=True)

        print(f"Fetched {len(df)} trades for {symbol}")

        if save:
            self.save_tick_to_parquet(df, symbol, data_type='tick')

        return df

    def resample_tick_to_ohlcv(
        self,
        tick_df: pd.DataFrame,
        timeframe: str = '30s'
    ) -> pd.DataFrame:
        """
        Resample tick data to OHLCV bars using pandas.

        Args:
            tick_df: Tick data DataFrame with timestamp index and 'price', 'amount' columns
            timeframe: Pandas resample frequency ('5s', '10s', '30s', '1min', etc.)

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if tick_df.empty:
            raise ValueError("Cannot resample empty tick data")

        if not isinstance(tick_df.index, pd.DatetimeIndex):
            raise ValueError("Tick data must have DatetimeIndex")

        # Resample price to OHLC
        ohlc = tick_df['price'].resample(timeframe).ohlc()

        # Resample amount to get volume
        volume = tick_df['amount'].resample(timeframe).sum()

        # Combine
        result = pd.concat([ohlc, volume], axis=1)

        # Rename volume column
        result.columns = ['open', 'high', 'low', 'close', 'volume']

        # Drop rows with no data (gaps in trading)
        result.dropna(subset=['open'], inplace=True)

        print(f"Resampled {len(tick_df)} ticks to {len(result)} {timeframe} bars")

        return result

    def fetch_and_resample_trades(
        self,
        symbol: str,
        resample_to: str = '30s',
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        hours_back: float = 1.0,
        save_raw: bool = False,
        save_resampled: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Fetch tick data and resample to OHLCV in one operation.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            resample_to: Pandas resample frequency ('5s', '10s', '30s', '1min', etc.)
            start_ms: Start timestamp in milliseconds (UTC)
            end_ms: End timestamp in milliseconds (UTC)
            hours_back: If start_ms not provided, fetch this many hours back
            save_raw: Whether to save raw tick data
            save_resampled: Whether to save resampled OHLCV data

        Returns:
            Tuple of (resampled_ohlcv, raw_tick_data or None)
        """
        # Fetch tick data
        tick_df = self.fetch_trades(
            symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            hours_back=hours_back,
            save=save_raw
        )

        # Resample to OHLCV
        ohlcv_df = self.resample_tick_to_ohlcv(tick_df, timeframe=resample_to)

        # Save resampled data
        if save_resampled:
            self.save_tick_to_parquet(ohlcv_df, symbol, data_type=f'{resample_to}_tick')

        # Return both if raw requested, otherwise just resampled
        raw_to_return = tick_df if save_raw else None

        return ohlcv_df, raw_to_return

    def save_tick_to_parquet(
        self,
        df: pd.DataFrame,
        symbol: str,
        data_type: str = 'tick'
    ) -> Path:
        """
        Save tick or resampled tick data to parquet with appropriate naming.

        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            data_type: 'tick' for raw, or '{timeframe}_tick' for resampled (e.g., '5S_tick')

        Returns:
            Path to saved file

        Naming convention:
            Raw tick: BTC_USDT_tick_20260420_20260420.parquet
            Resampled: BTC_USDT_5s_tick_20260420_20260420.parquet
        """
        clean_symbol = symbol.replace('/', '_')

        # Get date range from index
        start_date = df.index.min().strftime('%Y%m%d')
        end_date = df.index.max().strftime('%Y%m%d')

        # Create filename
        if data_type == 'tick':
            filename = f"{clean_symbol}_tick_{start_date}_{end_date}.parquet"
        else:
            # Extract timeframe from data_type (e.g., '5S_tick' -> '5s')
            timeframe = data_type.replace('_tick', '').lower()
            filename = f"{clean_symbol}_{timeframe}_tick_{start_date}_{end_date}.parquet"

        filepath = self.data_dir / filename
        df.to_parquet(filepath, compression='snappy')

        print(f"Saved to {filepath}")
        return filepath

    def list_available_data(self) -> List[str]:
        """
        List all available parquet files in data directory.

        Returns:
            List of filenames
        """
        parquet_files = list(self.data_dir.glob("*.parquet"))
        return [f.name for f in parquet_files]


def get_default_crypto_symbols() -> List[str]:
    """
    Get default list of crypto symbols for analysis.

    Returns:
        List of symbol strings (BTC, ETH, BNB, SOL, XRP vs USDT)
    """
    return [
        'BTC/USDT',
        'ETH/USDT',
        'BNB/USDT',
        'SOL/USDT',
        'XRP/USDT'
    ]


if __name__ == '__main__':
    # Example usage
    loader = BinanceDataLoader()

    # Fetch default crypto symbols for the past week
    symbols = get_default_crypto_symbols()
    data = loader.fetch_multiple_symbols(
        symbols=symbols,
        timeframe='1m',
        days_back=7,
        save=True
    )

    print(f"\nFetched data for {len(data)} symbols")
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")
