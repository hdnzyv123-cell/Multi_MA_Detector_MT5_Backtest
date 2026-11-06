"""MA/AP research backtest. No orders are sent. Python 3.10+, pandas, numpy.

Signals use closed bars; execution is next open. For ambiguous OHLC bars,
simulate both O-H-L-C and O-L-H-C and keep the lower closing equity outcome.
This is an explicit conservative approximation, not a tick reconstruction.
"""
from __future__ import annotations

import argparse
import copy
import json
import importlib
import math
import sys
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PORTFOLIO_MODES = [('M5', 'H1'), ('M15', 'H4'), ('H1', 'D1')]
TRADE_COLUMNS = ['entry_time', 'signal_time', 'exit_bar_time', 'direction',
                 'signal_type', 'entry', 'initial_stop', 'final_stop', 'target',
                 'lots', 'planned_risk', 'pnl', 'r_multiple', 'bars_held',
                 'partial_taken', 'exit_reason']


class SymbolInfo(Protocol):
    @property
    def point(self) -> float: ...
    @property
    def volume_min(self) -> float: ...
    @property
    def volume_step(self) -> float: ...
    @property
    def volume_max(self) -> float: ...
    @property
    def trade_contract_size(self) -> float: ...


class TickInfo(Protocol):
    @property
    def ask(self) -> float: ...


class AccountInfo(Protocol):
    @property
    def currency(self) -> str: ...


class MT5API(Protocol):
    """Typed boundary for the MT5 compiled module, which ships without stubs."""
    @property
    def ORDER_TYPE_BUY(self) -> int: ...
    def initialize(self) -> bool: ...
    def last_error(self) -> tuple[int, str]: ...
    def symbol_select(self, symbol: str, enable: bool) -> bool: ...
    def symbol_info(self, symbol: str) -> SymbolInfo | None: ...
    def symbol_info_tick(self, symbol: str) -> TickInfo | None: ...
    def account_info(self) -> AccountInfo | None: ...
    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int,
                            count: int) -> NDArray[np.void] | None: ...
    def copy_rates_range(self, symbol: str, timeframe: int, date_from: object,
                         date_to: object) -> NDArray[np.void] | None: ...
    def order_calc_profit(self, action: int, symbol: str, volume: float,
                          price_open: float, price_close: float) -> float | None: ...
    def shutdown(self) -> None: ...


def column(df: pd.DataFrame, name: str) -> pd.Series:
    """Scalar column selection must yield a Series, not duplicate-column frames."""
    result = df[name]
    if not isinstance(result, pd.Series):
        raise ValueError(f'Duplicate column name: {name}')
    return result


def bar_time(df: pd.DataFrame, index: int) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.DatetimeIndex(df.index)[index])


@dataclass(frozen=True)
class Config:
    symbol: str = 'XAUUSD'
    timeframe: str = 'M15'
    bars: int = 5760
    initial_equity: float = 10000.0
    risk_cash: float = 100.0
    risk_pct: float = 0.0  # When > 0, overrides risk_cash; 0.01 means 1%.
    ap_mode: str = 'rolling'  # rolling tick-volume weighted price or daily UTC
    ap_length: int = 50
    alma_length: int = 9
    kama_length: int = 21
    atr_length: int = 14
    breakout_atr: float = 0.3
    stop_lookback: int = 40
    stop_buffer_atr: float = 0.2
    min_stop_atr: float = 1.5
    target_r: float = 2.0
    target_mode: str = 'fixed'  # fixed or confirmed historical swing
    zone_lookback: int = 150
    partial_fraction: float = 0.4
    partial_r: float = 1.0
    breakeven: bool = True
    time_stop_bars: int = 144
    use_pullback: bool = False
    use_trend_filter: bool = False
    use_adx_filter: bool = False
    adx_threshold: float = 22.0
    use_volume_filter: bool = False
    use_mtf: bool = False
    higher_timeframe: str = 'D1'
    mtf_max_age_bars: int = 3
    min_volume_ratio: float = 0.7
    max_volume_ratio: float = 2.0
    session_start: int = 0  # UTC, next-open time, not signal-close time
    session_end: int = 24
    # CSV prices must be BID OHLC. spread is a price distance, never percent.
    spread_price: float = 0.30
    slippage_price: float = 0.05  # Market entry, stop, time/end exit, each fill
    pnl_per_price_unit_per_lot: float = 100.0  # USD account, 100 oz/lot example
    commission_per_lot_side: float = 0.0
    daily_carry_cost_per_lot: float = 0.0  # Constant UTC-day approximation
    lot_min: float = 0.01
    lot_step: float = 0.01
    lot_max: float = 100.0


def validate_config(cfg: Config):
    if cfg.initial_equity <= 0 or cfg.risk_cash <= 0 or not 0 <= cfg.risk_pct < 1:
        raise ValueError('Invalid equity/risk settings.')
    if min(cfg.ap_length, cfg.alma_length, cfg.kama_length, cfg.atr_length,
           cfg.stop_lookback, cfg.zone_lookback, cfg.time_stop_bars) < 1:
        raise ValueError('Lengths and time stop must be positive.')
    if not 0 <= cfg.partial_fraction < 1 or cfg.partial_r <= 0:
        raise ValueError('Invalid partial exit settings.')
    if cfg.target_r <= cfg.partial_r or cfg.min_stop_atr <= 0:
        raise ValueError('Target R must exceed partial R; minimum stop must be positive.')
    if min(cfg.spread_price, cfg.slippage_price, cfg.commission_per_lot_side,
           cfg.daily_carry_cost_per_lot, cfg.stop_buffer_atr, cfg.breakout_atr) < 0:
        raise ValueError('Costs and buffers cannot be negative.')
    if not 0 <= cfg.session_start <= 23 or not 0 <= cfg.session_end <= 24:
        raise ValueError('Invalid UTC session.')
    if cfg.session_start == cfg.session_end:
        raise ValueError('Session start and end cannot be equal.')
    if cfg.pnl_per_price_unit_per_lot <= 0 or cfg.lot_min <= 0 or cfg.lot_step <= 0 or cfg.lot_max < cfg.lot_min:
        raise ValueError('Invalid contract or volume settings.')
    if cfg.ap_mode not in ('rolling', 'daily') or cfg.target_mode not in ('fixed', 'zone'):
        raise ValueError('Invalid AP or target mode.')
    if cfg.min_volume_ratio < 0 or cfg.max_volume_ratio < cfg.min_volume_ratio:
        raise ValueError('Invalid volume ratio bounds.')
    if cfg.use_mtf and timeframe_minutes(cfg.higher_timeframe) <= timeframe_minutes(cfg.timeframe):
        raise ValueError('MTF onay zaman dilimi, giriş zaman diliminden büyük olmalıdır.')
    if cfg.mtf_max_age_bars < 1:
        raise ValueError('MTF veri geçerlilik süresi pozitif olmalıdır.')


def timeframe_minutes(name: str) -> int:
    supported = {'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4, 'M5': 5, 'M6': 6,
                 'M10': 10, 'M12': 12, 'M15': 15, 'M20': 20, 'M30': 30,
                 'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240, 'H6': 360,
                 'H8': 480, 'H12': 720, 'D1': 1440, 'W1': 10080}
    if name not in supported:
        raise ValueError(f'MTF için desteklenmeyen zaman dilimi: {name}')
    return supported[name]


def normalize_data(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = raw.copy()
    if 'time' in df:
        times = df.pop('time')
        df.index = pd.to_datetime(times, unit='s', utc=True) if pd.api.types.is_numeric_dtype(times) else pd.to_datetime(times, utc=True)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('A time column or DatetimeIndex is required.')
        df.index = pd.to_datetime(df.index, utc=True)
    if df.columns.has_duplicates:
        raise ValueError('Duplicate column names.')
    if df.index.has_duplicates or np.asarray(df.index.isna()).any():
        raise ValueError('Duplicate or missing timestamps.')
    df = df.sort_index()
    if 'volume' not in df and 'tick_volume' in df:
        df['volume'] = df['tick_volume']
    required = ['open', 'high', 'low', 'close', 'volume']
    if any(k not in df for k in required):
        raise ValueError(f'Required columns: time, {required}')
    for key in required:
        df[key] = pd.to_numeric(df[key], errors='raise')
    if not np.isfinite(df[required].to_numpy()).all() or (df['volume'] < 0).any():
        raise ValueError('Invalid prices/volume.')
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('Prices must be positive.')
    if ((df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
        (df['low'] > df[['open', 'close', 'high']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC.')
    if 'spread_price' not in df:
        df['spread_price'] = cfg.spread_price
    df['spread_price'] = pd.to_numeric(df['spread_price'], errors='raise')
    if not np.isfinite(df['spread_price']).all() or (df['spread_price'] < 0).any():
        raise ValueError('Invalid spread_price.')
    if len(df) < 2:
        raise ValueError('At least two closed bars are required.')
    return df


def fetch_mt5(cfg: Config) -> tuple[pd.DataFrame, dict]:
    mt5 = cast(MT5API, importlib.import_module('MetaTrader5'))
    if not mt5.initialize():
        raise RuntimeError(f'MT5 initialization failed: {mt5.last_error()}')
    try:
        if not mt5.symbol_select(cfg.symbol, True):
            raise RuntimeError(f'Symbol unavailable: {cfg.symbol}')
        timeframe = getattr(mt5, f'TIMEFRAME_{cfg.timeframe}', None)
        if not isinstance(timeframe, int):
            raise ValueError('Unsupported MT5 timeframe.')
        info = mt5.symbol_info(cfg.symbol)
        tick = mt5.symbol_info_tick(cfg.symbol)
        account = mt5.account_info()
        # Position 1 excludes the currently forming bar.
        rates = mt5.copy_rates_from_pos(cfg.symbol, timeframe, 1, cfg.bars)
        if rates is None or len(rates) < 2 or info is None or tick is None or account is None:
            raise RuntimeError(f'MT5 data/metadata unavailable: {mt5.last_error()}')
        factor = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, cfg.symbol, 1.0, tick.ask, tick.ask + 1.0)
        if factor is None or factor <= 0:
            raise RuntimeError('Cannot determine PnL conversion.')
        data = pd.DataFrame(rates)
        # MqlRates spread is points. Bar spread is only an OHLC approximation.
        data['spread_price'] = data['spread'] * info.point
        metadata = dict(pnl_per_price_unit_per_lot=float(factor), lot_min=info.volume_min,
                        lot_step=info.volume_step, lot_max=info.volume_max)
        return data, {'contract_config': metadata, 'currency': account.currency,
                      'requested_bars': cfg.bars, 'received_bars': len(data),
                      'point': info.point, 'contract_size': info.trade_contract_size,
                      'pnl_conversion': 'constant snapshot from order_calc_profit; historical FX conversion not modeled'}
    finally:
        mt5.shutdown()


def fetch_mt5_higher(cfg: Config, data: pd.DataFrame) -> pd.DataFrame:
    mt5 = cast(MT5API, importlib.import_module('MetaTrader5'))
    if not mt5.initialize():
        raise RuntimeError(f'MT5 initialization failed: {mt5.last_error()}')
    try:
        timeframe = getattr(mt5, f'TIMEFRAME_{cfg.higher_timeframe}', None)
        if not isinstance(timeframe, int):
            raise ValueError('Unsupported higher timeframe.')
        warmup = max(cfg.ap_length+cfg.kama_length+cfg.alma_length, 200)
        start = bar_time(data, 0).to_pydatetime() - timedelta(minutes=timeframe_minutes(cfg.higher_timeframe)*warmup*3)
        end = (bar_time(data, len(data)-1) + pd.Timedelta(minutes=timeframe_minutes(cfg.timeframe))).to_pydatetime()
        rates = mt5.copy_rates_range(cfg.symbol, timeframe, start, end)
        if rates is None or len(rates) < 2:
            raise RuntimeError(f'{cfg.higher_timeframe} MTF geçmişi alınamadı: {mt5.last_error()}')
        raw = pd.DataFrame(rates)
        info = mt5.symbol_info(cfg.symbol)
        if info is None:
            raise RuntimeError('MTF symbol metadata unavailable.')
        raw['spread_price'] = raw['spread']*info.point
        higher = normalize_data(raw, cfg)
        close_times = pd.DatetimeIndex(higher.index)+pd.Timedelta(minutes=timeframe_minutes(cfg.higher_timeframe))
        higher = higher.loc[close_times <= pd.Timestamp(end)].copy()
        if len(higher) < 2:
            raise ValueError('Yeterli kapanmış üst zaman dilimi mumu yok.')
        return higher
    finally:
        mt5.shutdown()


def fetch_mt5_portfolio(cfg: Config) -> tuple[dict[str, pd.DataFrame], dict, pd.Timestamp, pd.Timestamp]:
    """Use the M5 requested window for all engines; retrieve extra indicator history."""
    mt5 = cast(MT5API, importlib.import_module('MetaTrader5'))
    if not mt5.initialize():
        raise RuntimeError(f'MT5 initialization failed: {mt5.last_error()}')
    try:
        if not mt5.symbol_select(cfg.symbol, True):
            raise RuntimeError(f'Symbol unavailable: {cfg.symbol}')
        anchor_tf = getattr(mt5, 'TIMEFRAME_M5', None)
        if not isinstance(anchor_tf, int):
            raise RuntimeError('MT5 M5 timeframe unavailable.')
        anchor = mt5.copy_rates_from_pos(cfg.symbol, anchor_tf, 1, cfg.bars)
        info, tick, account = mt5.symbol_info(cfg.symbol), mt5.symbol_info_tick(cfg.symbol), mt5.account_info()
        if anchor is None or len(anchor) < 2 or info is None or tick is None or account is None:
            raise RuntimeError(f'Portfolio data unavailable: {mt5.last_error()}')
        factor = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, cfg.symbol, 1., tick.ask, tick.ask+1.)
        if factor is None or factor <= 0:
            raise RuntimeError('Cannot determine contract PnL conversion.')
        anchor_data = normalize_data(pd.DataFrame(anchor), cfg)
        start = bar_time(anchor_data, 0)
        end = bar_time(anchor_data, len(anchor_data)-1)+pd.Timedelta(minutes=5)
        warmup = max(cfg.ap_length+cfg.kama_length+cfg.alma_length, 200)
        frames: dict[str, pd.DataFrame] = {}
        for tf in ('M5', 'M15', 'H1', 'H4', 'D1'):
            tf_id = getattr(mt5, f'TIMEFRAME_{tf}', None)
            if not isinstance(tf_id, int):
                raise RuntimeError(f'MT5 timeframe unavailable: {tf}')
            history_start = start.to_pydatetime()-timedelta(minutes=timeframe_minutes(tf)*warmup*3)
            rates = mt5.copy_rates_range(cfg.symbol, tf_id, history_start, end.to_pydatetime())
            if rates is None or len(rates) < 2:
                raise RuntimeError(f'{tf} history unavailable: {mt5.last_error()}')
            raw = pd.DataFrame(rates)
            raw['spread_price'] = raw['spread']*info.point
            frame = normalize_data(raw, cfg)
            closed_at = pd.DatetimeIndex(frame.index)+pd.Timedelta(minutes=timeframe_minutes(tf))
            frames[tf] = frame.loc[closed_at <= end].copy()
        metadata = {'source': 'MT5', 'currency': account.currency,
                    'contract_config': dict(pnl_per_price_unit_per_lot=float(factor),
                        lot_min=info.volume_min, lot_step=info.volume_step, lot_max=info.volume_max),
                    'requested_m5_bars': cfg.bars,
                    'pnl_conversion': 'constant current order_calc_profit snapshot'}
        return frames, metadata, start, end
    finally:
        mt5.shutdown()


def alma(s: pd.Series, length: int) -> pd.Series:
    weights = np.exp(-((np.arange(length) - 0.85 * (length - 1)) ** 2) / (2 * (length / 6.0) ** 2))
    weights /= weights.sum()
    return pd.Series(s.rolling(length).apply(lambda x: np.dot(x, weights), raw=True), index=s.index)


def kama(s: pd.Series, length: int) -> pd.Series:
    volatility = pd.Series(s.diff().abs().rolling(length).sum(), index=s.index)
    er = pd.Series(s.diff(length).abs() / volatility.replace(0, np.nan), index=s.index).fillna(0)
    smooth = (er * (2 / 3 - 2 / 31) + 2 / 31) ** 2
    out = pd.Series(np.nan, index=s.index)
    value = np.nan
    for i in range(len(s)):
        if i < length or not np.isfinite(s.iloc[i]) or not np.isfinite(s.iloc[i-length:i+1]).all():
            continue
        value = s.iloc[i] if not np.isfinite(value) else value + smooth.iloc[i] * (s.iloc[i] - value)
        out.iloc[i] = value
    return out


def build_features(data: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = data.copy()
    c, h, l, v = (column(df, k) for k in ('close', 'high', 'low', 'volume'))
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/cfg.atr_length, adjust=False, min_periods=cfg.atr_length).mean()
    if cfg.ap_mode == 'daily':
        day = [bar_time(df, i).floor('D') for i in range(len(df))]
        denominator = v.groupby(day).cumsum().replace(0, np.nan)
        df['ap'] = (c*v).groupby(day).cumsum() / denominator
    else:
        volume_sum = pd.Series(v.rolling(cfg.ap_length).sum(), index=df.index)
        df['ap'] = (c*v).rolling(cfg.ap_length).sum() / volume_sum.replace(0, np.nan)
    df['alma'] = alma(column(df, 'ap'), cfg.alma_length)
    df['kama'] = kama(column(df, 'ap'), cfg.kama_length)
    up, down = h.diff(), -l.diff()
    plus = up.where((up > down) & (up > 0), 0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    minus = down.where((down > up) & (down > 0), 0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    denominator = pd.Series(plus + minus, index=df.index).replace(0, np.nan)
    df['adx'] = (100 * (plus-minus).abs() / denominator).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    volume_mean = pd.Series(v.rolling(20).mean(), index=df.index)
    df['volume_ratio'] = v / volume_mean.replace(0, np.nan)
    fast, slow = c.rolling(50).mean(), c.rolling(200).mean()
    long = (df['alma'] > df['kama']) & (df['alma'].shift() <= df['kama'].shift()) & ((c-df['kama']) >= cfg.breakout_atr*df['atr'])
    short = (df['alma'] < df['kama']) & (df['alma'].shift() >= df['kama'].shift()) & ((df['kama']-c) >= cfg.breakout_atr*df['atr'])
    tags = pd.Series('BREAKOUT', index=df.index)
    if cfg.use_pullback:
        pb_long = (c.shift() > df['ap'].shift()) & (l <= df['ap']) & (c > df['ap']) & (df['alma'] > df['kama'])
        pb_short = (c.shift() < df['ap'].shift()) & (h >= df['ap']) & (c < df['ap']) & (df['alma'] < df['kama'])
        tags.loc[(pb_long & ~long) | (pb_short & ~short)] = 'PULLBACK'
        long, short = long | pb_long, short | pb_short
    if cfg.use_trend_filter:
        long &= (c > fast) & (fast > slow)
        short &= (c < fast) & (fast < slow)
    if cfg.use_adx_filter:
        long &= df['adx'] >= cfg.adx_threshold
        short &= df['adx'] >= cfg.adx_threshold
    if cfg.use_volume_filter:
        allowed = df['volume_ratio'].between(cfg.min_volume_ratio, cfg.max_volume_ratio)
        long, short = long & allowed, short & allowed
    ready = df[['ap', 'alma', 'kama', 'atr']].notna().all(axis=1) & (df['atr'] > 0)
    df['signal'] = np.select([long & ~short & ready, short & ~long & ready], [1, -1], default=0)
    df['signal_type'] = tags
    return df


def apply_mtf_filter(features: pd.DataFrame, higher: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Join HTF state by its CLOSE time, never by its opening timestamp."""
    validate_config(cfg)
    if not cfg.use_mtf:
        return features.copy()
    if len(higher) < 2:
        raise ValueError('MTF requires at least two higher-timeframe bars.')
    df = features.copy()
    # Only indicators are used from HTF; its entry filters do not define trend.
    upper_cfg = Config(**{**asdict(cfg), 'use_mtf': False,
                          'use_trend_filter': False, 'use_adx_filter': False,
                          'use_volume_filter': False, 'use_pullback': False})
    upper = build_features(higher, upper_cfg)
    upper_index = pd.DatetimeIndex(upper.index)
    upper_ready = upper[['ap', 'alma', 'kama']].notna().all(axis=1)
    bull = (upper['alma'] > upper['kama']) & (upper['close'] > upper['ap']) & upper_ready
    bear = (upper['alma'] < upper['kama']) & (upper['close'] < upper['ap']) & upper_ready
    state = pd.DataFrame({'mtf_bar_open': upper_index,
        'mtf_available_at': upper_index+pd.Timedelta(minutes=timeframe_minutes(cfg.higher_timeframe)),
        'mtf_direction': np.select([bull, bear], [1, -1], default=0),
        'mtf_ready': upper_ready.to_numpy(), 'mtf_close': column(upper, 'close').to_numpy(),
        'mtf_ap': column(upper, 'ap').to_numpy(), 'mtf_alma': column(upper, 'alma').to_numpy(),
        'mtf_kama': column(upper, 'kama').to_numpy()})
    decisions = pd.DataFrame({'decision_time': pd.DatetimeIndex(df.index)+pd.Timedelta(minutes=timeframe_minutes(cfg.timeframe))})
    aligned = pd.merge_asof(decisions, state.sort_values('mtf_available_at'),
                            left_on='decision_time', right_on='mtf_available_at', direction='backward',
                            tolerance=pd.Timedelta(minutes=timeframe_minutes(cfg.higher_timeframe)*cfg.mtf_max_age_bars))
    for key in ('mtf_bar_open', 'mtf_available_at', 'mtf_ready', 'mtf_close', 'mtf_ap', 'mtf_alma', 'mtf_kama'):
        df[key] = column(aligned, key).to_numpy()
    df['mtf_direction'] = column(aligned, 'mtf_direction').fillna(0).astype(int).to_numpy()
    df['mtf_ready'] = column(aligned, 'mtf_ready').eq(True).to_numpy()
    df['signal_before_mtf'] = column(df, 'signal').copy()
    df['mtf_blocked'] = (df['signal'] != 0) & ((df['signal'] != df['mtf_direction']) | ~df['mtf_ready'])
    df.loc[df['mtf_blocked'], 'signal'] = 0
    return df


@dataclass
class Position:
    direction: int
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
    signal_type: str
    entry: float
    initial_stop: float
    stop: float
    target: float
    partial_target: float
    initial_qty: float
    remaining: float
    planned_risk: float
    initial_distance: float
    realized: float
    bars: int = 0
    partial_taken: bool = False
    partial_filled: bool = False
    last_day: pd.Timestamp | None = None


def in_session(t: pd.Timestamp, cfg: Config) -> bool:
    if cfg.session_start < cfg.session_end:
        return cfg.session_start <= t.hour < cfg.session_end
    return t.hour >= cfg.session_start or t.hour < cfg.session_end


def round_lots(qty: float, cfg: Config) -> float:
    return round(math.floor((min(qty, cfg.lot_max)+1e-12)/cfg.lot_step)*cfg.lot_step, 10)


def open_position(df: pd.DataFrame, i: int, equity: float, cfg: Config) -> tuple[Position | None, str]:
    prev, bar = df.iloc[i-1], df.iloc[i]
    direction = int(prev['signal'])
    if not direction or not in_session(bar_time(df, i), cfg):
        return None, 'NO_SIGNAL_OR_SESSION'
    atr = float(prev['atr'])
    if not np.isfinite(atr) or atr <= 0 or equity <= 0:
        return None, 'INVALID_ATR_OR_EQUITY'
    entry = float(bar['open']) + (float(bar['spread_price']) if direction == 1 else 0) + direction*cfg.slippage_price
    history = df.iloc[max(0, i-cfg.stop_lookback):i]
    stop = float(history['low'].min()-cfg.stop_buffer_atr*atr) if direction == 1 else float(history['high'].max()+cfg.stop_buffer_atr*atr)
    distance = max(direction*(entry-stop), cfg.min_stop_atr*atr)
    stop = entry-direction*distance
    risk_budget = equity*cfg.risk_pct if cfg.risk_pct else cfg.risk_cash
    per_lot_risk = (distance+cfg.slippage_price)*cfg.pnl_per_price_unit_per_lot + 2*cfg.commission_per_lot_side
    qty = round_lots(risk_budget/per_lot_risk, cfg)
    if qty < cfg.lot_min:
        return None, 'BELOW_MIN_LOT'
    # Partial fills must respect the broker's minimum and step too.
    partial_qty = round_lots(qty*cfg.partial_fraction, cfg)
    be_offset = 2*cfg.commission_per_lot_side/cfg.pnl_per_price_unit_per_lot + cfg.slippage_price
    if cfg.breakeven and partial_qty >= cfg.lot_min and be_offset >= distance*cfg.partial_r:
        return None, 'COSTS_TOO_HIGH_FOR_BREAKEVEN'
    target = entry + direction*distance*cfg.target_r
    if cfg.target_mode == 'zone':
        candidates = []
        for j in range(max(2, i-cfg.zone_lookback), i-2):
            level = float(df.iloc[j]['high' if direction == 1 else 'low'])
            neighbors = df.iloc[j-2:j+3]['high' if direction == 1 else 'low']
            confirmed = level == float(neighbors.max() if direction == 1 else neighbors.min())
            intervening = df.iloc[j+1:i]['high' if direction == 1 else 'low']
            untouched = (intervening < level).all() if direction == 1 else (intervening > level).all()
            if confirmed and untouched and direction*(level-entry) >= cfg.target_r*distance:
                candidates.append(level)
        if not candidates:
            return None, 'NO_VALID_ZONE'
        target = min(candidates) if direction == 1 else max(candidates)
    pos = Position(direction, bar_time(df, i), bar_time(df, i-1), str(prev['signal_type']), entry,
                   stop, stop, target, entry+direction*distance*cfg.partial_r,
                   qty, qty, per_lot_risk*qty, distance, -qty*cfg.commission_per_lot_side,
                   last_day=bar_time(df, i).floor('D'))
    # Disable partial when either leg would be below minimum lot.
    if partial_qty < cfg.lot_min or qty-partial_qty < cfg.lot_min:
        pos.partial_taken = True
    return pos, ''


def close_qty(pos: Position, price: float, qty: float, cfg: Config):
    pos.realized += pos.direction*(price-pos.entry)*cfg.pnl_per_price_unit_per_lot*qty - cfg.commission_per_lot_side*qty
    pos.remaining = max(0.0, round(pos.remaining-qty, 10))


def take_partial(pos: Position, cfg: Config):
    qty = round_lots(pos.initial_qty*cfg.partial_fraction, cfg)
    close_qty(pos, pos.partial_target, qty, cfg)
    pos.partial_taken = True
    pos.partial_filled = True
    if cfg.breakeven:
        # Remaining leg breaks even after both commissions and stop slippage,
        # assuming no gap. Accrued carry is reported separately in total PnL.
        pos.stop = pos.entry + pos.direction*(2*cfg.commission_per_lot_side/cfg.pnl_per_price_unit_per_lot + cfg.slippage_price)


def simulate_path(original: Position, points: list[float], cfg: Config) -> tuple[Position, str]:
    pos = copy.deepcopy(original)
    direction = pos.direction
    opening = points[0]  # Already expressed in the executable exit-side quote.
    if direction*(opening-pos.stop) <= 0:
        close_qty(pos, opening-direction*cfg.slippage_price, pos.remaining, cfg)
        return pos, 'STOP_GAP'
    if direction*(opening-pos.target) >= 0:
        if not pos.partial_taken:
            take_partial(pos, cfg)
        close_qty(pos, pos.target, pos.remaining, cfg)  # Conservative limit fill
        return pos, 'TARGET'
    if not pos.partial_taken and direction*(opening-pos.partial_target) >= 0:
        take_partial(pos, cfg)
    for start, end in zip(points, points[1:]):
        if direction*(end-start) < 0:
            if min(start, end) <= pos.stop <= max(start, end):
                close_qty(pos, pos.stop-direction*cfg.slippage_price, pos.remaining, cfg)
                return pos, 'BE_STOP' if pos.partial_taken and cfg.breakeven and pos.stop != pos.initial_stop else 'STOP'
        else:
            if not pos.partial_taken and min(start, end) <= pos.partial_target <= max(start, end):
                take_partial(pos, cfg)
            if min(start, end) <= pos.target <= max(start, end):
                close_qty(pos, pos.target, pos.remaining, cfg)
                return pos, 'TARGET'
    return pos, ''


def mark_pnl(pos: Position, quote: float, cfg: Config) -> float:
    return pos.realized + pos.direction*(quote-pos.entry)*cfg.pnl_per_price_unit_per_lot*pos.remaining - cfg.commission_per_lot_side*pos.remaining


def backtest(df: pd.DataFrame, cfg: Config, start_at: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    validate_config(cfg)
    if not 0 <= start_at < len(df):
        raise ValueError('Evaluation start lies outside the data.')
    records, curve, skipped = [], [], {}
    balance = cfg.initial_equity
    pos = None
    for i in range(start_at, len(df)):
        bar, timestamp = df.iloc[i], bar_time(df, i)
        # At most one new entry per bar, and only at its open.
        if pos is None and i > 0:
            pos, reason = open_position(df, i, balance, cfg)
            if reason != 'NO_SIGNAL_OR_SESSION' and pos is None:
                skipped[reason] = skipped.get(reason, 0)+1
        exposed_this_bar = pos is not None
        if pos is not None:
            day = timestamp.floor('D')
            days = (day-(pos.last_day if pos.last_day is not None else day)).days
            pos.realized -= max(0, days)*cfg.daily_carry_cost_per_lot*pos.remaining
            pos.last_day = day
            pos.bars += 1
            offset = float(bar['spread_price']) if pos.direction == -1 else 0.0
            op, hi, lo, cl = (float(bar[k])+offset for k in ('open', 'high', 'low', 'close'))
            paths = [simulate_path(pos, [op, hi, lo, cl], cfg), simulate_path(pos, [op, lo, hi, cl], cfg)]
            pos, exit_reason = min(paths, key=lambda result: mark_pnl(result[0], cl, cfg))
            if pos.remaining > 0 and (pos.bars >= cfg.time_stop_bars or i == len(df)-1):
                exit_reason = 'END_OF_DATA' if i == len(df)-1 else 'TIME_STOP'
                close_qty(pos, cl-pos.direction*cfg.slippage_price, pos.remaining, cfg)
            if pos.remaining <= 0:
                records.append(dict(entry_time=pos.entry_time, signal_time=pos.signal_time,
                    exit_bar_time=timestamp, direction='LONG' if pos.direction == 1 else 'SHORT',
                    signal_type=pos.signal_type, entry=pos.entry, initial_stop=pos.initial_stop,
                    final_stop=pos.stop, target=pos.target, lots=pos.initial_qty,
                    planned_risk=pos.planned_risk, pnl=pos.realized,
                    r_multiple=pos.realized/pos.planned_risk, bars_held=pos.bars,
                    partial_taken=pos.partial_filled,
                    exit_reason=exit_reason))
                balance += pos.realized
                pos = None
        unrealized = 0.0 if pos is None else mark_pnl(pos, float(bar['close'])+(float(bar['spread_price']) if pos.direction == -1 else 0), cfg)
        curve.append(dict(time=timestamp, balance=balance, equity=balance+unrealized, in_position=pos is not None, exposed_this_bar=exposed_this_bar))
    trades = pd.DataFrame(records, columns=TRADE_COLUMNS)
    equity = pd.DataFrame(curve).set_index('time')
    equity_values = column(equity, 'equity')
    peak = equity_values.cummax().clip(lower=cfg.initial_equity)
    equity['drawdown_cash'] = peak-equity['equity']
    equity['drawdown_pct'] = equity['drawdown_cash']/peak*100
    pnl = column(trades, 'pnl') if not trades.empty else pd.Series(dtype=float)
    gross_win, gross_loss = float(pnl[pnl > 1e-8].sum()), float(-pnl[pnl < -1e-8].sum())
    summary = dict(closed_trades=len(trades), wins=int((pnl > 1e-8).sum()),
        breakevens=int((pnl.abs() <= 1e-8).sum()), losses=int((pnl < -1e-8).sum()),
        win_rate_pct=float((pnl > 1e-8).mean()*100) if len(pnl) else None,
        profit_factor=gross_win/gross_loss if gross_loss else None,
        net_pnl=float(pnl.sum()), expectancy_cash=float(pnl.mean()) if len(pnl) else None,
        expectancy_r=float(column(trades, 'r_multiple').mean()) if len(pnl) else None,
        final_equity=float(equity_values.iloc[-1]),
        return_pct=(float(equity_values.iloc[-1])/cfg.initial_equity-1)*100,
        max_bar_close_drawdown_cash=float(column(equity, 'drawdown_cash').max()),
        max_bar_close_drawdown_pct=float(column(equity, 'drawdown_pct').max()),
        exposure_pct=float(column(equity, 'exposed_this_bar').mean()*100),
        start=str(df.index[start_at]), end=str(df.index[-1]), actual_bars=len(df)-start_at,
        skipped_entries=skipped, exit_counts=trades['exit_reason'].value_counts().to_dict() if len(pnl) else {})
    if cfg.use_mtf:
        evaluated = df.iloc[start_at:]
        if 'mtf_blocked' not in evaluated:
            raise ValueError('MTF açık ama üst zaman dilimi filtresi uygulanmamış.')
        before = column(evaluated, 'signal_before_mtf').ne(0)
        summary['mtf'] = dict(higher_timeframe=cfg.higher_timeframe,
            signals_before=int(before.sum()), signals_after=int(column(evaluated, 'signal').ne(0).sum()),
            blocked_signals=int(column(evaluated, 'mtf_blocked').sum()),
            no_ready_htf_signals=int((before & ~column(evaluated, 'mtf_ready')).sum()))
    return trades, equity, summary


def display_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None:
        return 'Hesaplanamadı'
    return f'{value:,.{decimals}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def format_report(summary: dict, cfg: Config, currency: str) -> str:
    unit = currency if currency != 'configured account currency' else 'hesap para birimi'
    lines = ['=' * 68, 'MA / AP BACKTEST RAPORU', '=' * 68]

    def section(title: str):
        lines.extend(['', title, '-' * 68])

    def row(label: str, value: str):
        lines.append(f'{label:<36}: {value}')

    def money(value: float | int | None) -> str:
        return f'{display_number(value)} {unit}' if value is not None else 'Hesaplanamadı'

    section('1. VERİ VE STRATEJİ')
    row('Sembol / zaman dilimi', f'{cfg.symbol} / {cfg.timeframe}')
    row('Başlangıç (UTC)', str(summary['start']).replace('+00:00', ''))
    row('Bitiş (UTC)', str(summary['end']).replace('+00:00', ''))
    row('İncelenen mum', display_number(summary['actual_bars'], 0))
    row('AP yöntemi', f'{cfg.ap_length} mum hacim ağırlıklı ortalama' if cfg.ap_mode == 'rolling' else 'Günlük hacim ağırlıklı ortalama (UTC)')
    filters = [name for name, enabled in [('Trend', cfg.use_trend_filter), ('ADX', cfg.use_adx_filter),
               ('Hacim', cfg.use_volume_filter), ('Pullback', cfg.use_pullback)] if enabled]
    row('Ek filtreler / girişler', ', '.join(filters) if filters else 'Kapalı; MA/AP çekirdeği')
    row('MTF trend onayı', f'{cfg.timeframe} giriş + {cfg.higher_timeframe} onay' if cfg.use_mtf else 'Kapalı')
    if cfg.use_mtf:
        mtf = summary.get('mtf', {})
        row('MTF öncesi / sonrası sinyal', f"{mtf.get('signals_before', 0)} / {mtf.get('signals_after', 0)}")
        row('MTF nedeniyle elenen sinyal', display_number(mtf.get('blocked_signals', 0), 0))
        row('Üst veri / ısınma eksik sinyal', display_number(mtf.get('no_ready_htf_signals', 0), 0))

    section('2. İŞLEM SONUÇLARI')
    row('Toplam kapanan işlem', display_number(summary['closed_trades'], 0))
    row('Kârlı / zararlı / nötr', f"{summary['wins']} / {summary['losses']} / {summary['breakevens']}")
    rate = summary['win_rate_pct']
    row('Kazanma oranı', f'%{display_number(rate)}' if rate is not None else 'İşlem yok')

    section('3. KÂR VE GETİRİ')
    row('Başlangıç sermayesi', money(cfg.initial_equity))
    row('Son sermaye', money(summary['final_equity']))
    row('Toplam net kâr / zarar', money(summary['net_pnl']))
    row('Toplam getiri', f"%{display_number(summary['return_pct'])}")
    row('Kâr faktörü', display_number(summary['profit_factor']))
    row('İşlem başına ortalama sonuç', money(summary['expectancy_cash']))
    r_value = summary['expectancy_r']
    row('İşlem başına ortalama R', f'{display_number(r_value)} R' if r_value is not None else 'İşlem yok')

    section('4. RİSK VE POZİSYON SÜRESİ')
    row('İşlem başına risk bütçesi', f'%{display_number(cfg.risk_pct*100)} sermaye' if cfg.risk_pct else money(cfg.risk_cash))
    row('En büyük sermaye düşüşü', money(summary['max_bar_close_drawdown_cash']))
    row('En büyük düşüş oranı', f"%{display_number(summary['max_bar_close_drawdown_pct'])}")
    row('Pozisyon bulunan mum oranı', f"%{display_number(summary['exposure_pct'])}")
    row('Zaman stopu', f'{cfg.time_stop_bars} mum')
    lines.extend(['  Düşüş ölçümü mum kapanışlarına göredir.', '  R: işlemin başlangıçta planlanan parasal riski.'])

    section('5. ÇIKIŞ NEDENLERİ')
    exit_labels = {'STOP': 'Stop loss', 'STOP_GAP': 'Fiyat boşluğunda stop',
                   'BE_STOP': 'Kısmi kâr sonrası BE stop', 'TARGET': 'Hedefe ulaşma',
                   'TIME_STOP': 'Zaman stopu', 'END_OF_DATA': 'Veri sonunda kapanış'}
    counts = summary['exit_counts']
    if counts:
        for key, count in counts.items():
            row(exit_labels.get(str(key), str(key)), display_number(count, 0))
    else:
        lines.append('  Kapanan işlem yok.')

    section('6. ELENEN GİRİŞLER')
    skip_labels = {'BELOW_MIN_LOT': 'Minimum lotun altında kalan',
                   'INVALID_ATR_OR_EQUITY': 'Geçersiz ATR veya sermaye',
                   'NO_VALID_ZONE': 'Uygun swing hedefi bulunamayan',
                   'COSTS_TOO_HIGH_FOR_BREAKEVEN': 'BE için maliyeti fazla yüksek'}
    skipped = summary['skipped_entries']
    if skipped:
        for key, count in skipped.items():
            row(skip_labels.get(str(key), str(key)), display_number(count, 0))
    else:
        lines.append('  Ek risk / hedef kuralları nedeniyle elenen giriş yok.')
    lines.extend(['  Bu sayılar seans dışında kalan veya açık pozisyon sırasında',
                  '  uygulanmayan sinyalleri içermez.', '', '=' * 68])
    return '\n'.join(lines)


def format_comparison(comparisons: list[dict]) -> str:
    names = {'core': 'MA/AP çekirdeği', 'trend': '+ Trend filtresi', 'adx': '+ ADX filtresi',
             'volume': '+ Hacim filtresi', 'pullback': '+ Pullback girişleri', 'without_partial': 'Kısmi çıkış kapalı', 'mtf': '+ MTF trend onayı'}
    lines = ['', '7. STRATEJİ KARŞILAŞTIRMASI', '-' * 82,
             f"{'Sürüm':<24} {'İşlem':>6} {'Kaz.%':>8} {'Net sonuç':>12} {'Kâr F.':>9} {'Düşüş %':>10}"]
    for result in comparisons:
        pf = result['profit_factor']
        pf_text = display_number(pf) if pf is not None else '-'
        rate = result['win_rate_pct']
        rate_text = display_number(rate) if rate is not None else '-'
        lines.append(f"{names.get(result['variant'], result['variant']):<24} "
                     f"{result['closed_trades']:>6} {rate_text:>8} "
                     f"{display_number(result['net_pnl']):>12} {pf_text:>9} "
                     f"{display_number(result['max_bar_close_drawdown_pct']):>10}")
    lines.extend(['  Net sonuç hesap para birimindedir; getiri/düşüş yıllıklandırılmamıştır.',
                  '  Sürümler, aynı çıkış ve risk ayarlarıyla çekirdeğe karşı ölçülür.',
                  '  Farklı zaman dilimi modlarını yalnızca ortak tarih aralığında karşılaştırın.'])
    return '\n'.join(lines)


def combine_portfolio(results: dict, cfg: Config, start: pd.Timestamp,
                      end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sum separately allocated engine equities at CLOSED-bar timestamps."""
    curves = []
    trade_tables = []
    for engine, result in results.items():
        equity = result['equity']
        engine_cfg = result['config']
        series = column(equity, 'equity').copy()
        series.index = pd.DatetimeIndex(series.index)+pd.Timedelta(minutes=timeframe_minutes(engine_cfg.timeframe))
        series.name = engine
        seed = pd.Series([engine_cfg.initial_equity], index=pd.DatetimeIndex([start]), name=engine)
        series = pd.concat([seed, series])
        series = series.loc[~series.index.duplicated(keep='last')].sort_index()
        curves.append(series)
        trades = result['trades'].copy()
        if not trades.empty:
            trades['engine'] = engine
            trades['entry_timeframe'] = engine_cfg.timeframe
            trades['confirmation_timeframe'] = engine_cfg.higher_timeframe
            trades['exit_bar_close_time'] = pd.to_datetime(column(trades, 'exit_bar_time'), utc=True)+pd.Timedelta(minutes=timeframe_minutes(engine_cfg.timeframe))
            trade_tables.append(trades)
    # Forward fill only; a future H1 or M15 close never appears in an earlier M5 mark.
    combined = pd.concat(curves, axis=1).sort_index().ffill()
    combined['equity'] = combined.sum(axis=1)
    peak = column(combined, 'equity').cummax().clip(lower=cfg.initial_equity)
    combined['drawdown_cash'] = peak-column(combined, 'equity')
    combined['drawdown_pct'] = column(combined, 'drawdown_cash')/peak*100
    trades = pd.concat(trade_tables, ignore_index=True).sort_values('entry_time').reset_index(drop=True) if trade_tables else pd.DataFrame(columns=TRADE_COLUMNS+['engine', 'entry_timeframe', 'confirmation_timeframe', 'exit_bar_close_time'])
    pnl = column(trades, 'pnl') if not trades.empty else pd.Series(dtype=float)
    gains, losses = float(pnl[pnl > 1e-8].sum()), float(-pnl[pnl < -1e-8].sum())
    summary = dict(closed_trades=len(trades), wins=int((pnl > 1e-8).sum()),
        losses=int((pnl < -1e-8).sum()), breakevens=int((pnl.abs() <= 1e-8).sum()),
        win_rate_pct=float((pnl > 1e-8).mean()*100) if len(pnl) else None,
        profit_factor=gains/losses if losses else None, net_pnl=float(pnl.sum()),
        expectancy_cash=float(pnl.mean()) if len(pnl) else None,
        expectancy_r=float(column(trades, 'r_multiple').mean()) if len(pnl) else None,
        initial_equity=cfg.initial_equity, final_equity=float(column(combined, 'equity').iloc[-1]),
        return_pct=(float(column(combined, 'equity').iloc[-1])/cfg.initial_equity-1)*100,
        max_sampled_drawdown_cash=float(column(combined, 'drawdown_cash').max()),
        max_sampled_drawdown_pct=float(column(combined, 'drawdown_pct').max()),
        start=str(start), end=str(end),
        exit_counts=column(trades, 'exit_reason').value_counts().to_dict() if len(pnl) else {})
    return trades, combined, summary


def portfolio_report(results: dict, summary: dict, cfg: Config, currency: str) -> str:
    lines = ['='*84, 'ÜÇ BAĞIMSIZ MTF MOTORU / TOPLAM PORTFÖY', '='*84,
             f"Ortak dönem (UTC): {summary['start']} -> {summary['end']}",
             f'Toplam başlangıç sermayesi: {display_number(cfg.initial_equity)} {currency}',
             'Sermaye üç motora eşit ayrılır; her motor aynı anda bir pozisyon tutabilir.',
             'M5/H1, M15/H4 ve H1/D1 motorları aynı anda bağımsız işlem açabilir.', '']
    lines.append(f"{'Giriş / onay':<18} {'İşlem':>6} {'Kaz.%':>8} {'Net sonuç':>13} {'Kâr F.':>9} {'Düşüş %':>10}")
    for name, result in results.items():
        row = result['summary']
        pf = display_number(row['profit_factor']) if row['profit_factor'] is not None else '-'
        rate = display_number(row['win_rate_pct']) if row['win_rate_pct'] is not None else '-'
        lines.append(f"{name:<18} {row['closed_trades']:>6} {rate:>8} {display_number(row['net_pnl']):>13} {pf:>9} {display_number(row['max_bar_close_drawdown_pct']):>10}")
    pf = display_number(summary['profit_factor']) if summary['profit_factor'] is not None else '-'
    rate = display_number(summary['win_rate_pct']) if summary['win_rate_pct'] is not None else '-'
    lines.extend(['-'*84,
        f"{'TOPLAM PORTFÖY':<18} {summary['closed_trades']:>6} {rate:>8} {display_number(summary['net_pnl']):>13} {pf:>9} {display_number(summary['max_sampled_drawdown_pct']):>10}",
        '', 'TOPLAM SONUÇLAR', '-'*84,
        f"Kârlı / zararlı / nötr     : {summary['wins']} / {summary['losses']} / {summary['breakevens']}",
        f"Son sermaye               : {display_number(summary['final_equity'])} {currency}",
        f"Toplam getiri             : %{display_number(summary['return_pct'])}",
        f"Ortalama işlem sonucu     : {display_number(summary['expectancy_cash'])} {currency}",
        f"Ortalama R                : {display_number(summary['expectancy_r'])}",
        f"En büyük toplam düşüş     : {display_number(summary['max_sampled_drawdown_cash'])} {currency}",
        'Toplam düşüş, eşleştirilen kapanış sermaye eğrisinden hesaplanır.',
        'Kazanma oranı ve kâr faktörü bütün işlemlerden yeniden hesaplanır.'])
    if not cfg.risk_pct:
        lines.append(f'İşlem risk bütçesi: motor başına {display_number(cfg.risk_cash)} {currency}; üç eşzamanlı girişte {display_number(cfg.risk_cash*3)} {currency}.')
    else:
        lines.append(f'İşlem riski: her motorun ayrılmış sermayesinin %{display_number(cfg.risk_pct*100)} kadarı.')
    for name, result in results.items():
        lines.extend(['', f'>>> {name} MOTORUNUN AYRINTILARI',
                      format_report(result['summary'], result['config'], currency)])
    return '\n'.join(lines)


def run_portfolio(cfg: Config, args: argparse.Namespace):
    validate_config(cfg)
    if args.data_dir:
        frames = {tf: normalize_data(pd.read_csv(args.data_dir/f'{tf}.csv'), cfg)
                  for tf in ('M5', 'M15', 'H1', 'H4', 'D1')}
        metadata = {'source': 'CSV', 'currency': 'hesap para birimi'}
        window_start = max(bar_time(frames[tf], 0) for tf, _ in PORTFOLIO_MODES)
        window_end = min(bar_time(frames[tf], len(frames[tf])-1)+pd.Timedelta(minutes=timeframe_minutes(tf)) for tf, _ in PORTFOLIO_MODES)
    else:
        frames, metadata, window_start, window_end = fetch_mt5_portfolio(cfg)
        cfg = Config(**{**asdict(cfg), **metadata['contract_config']})
        window_start = max(window_start, *(bar_time(frames[tf], 0) for tf, _ in PORTFOLIO_MODES))
        window_end = min(window_end, *(bar_time(frames[tf], len(frames[tf])-1)+pd.Timedelta(minutes=timeframe_minutes(tf)) for tf, _ in PORTFOLIO_MODES))
    if args.test_start:
        window_start = max(window_start, pd.to_datetime(args.test_start, utc=True))
    if window_start >= window_end:
        raise ValueError('Üç motor için ortak değerlendirme aralığı bulunamadı.')
    output = args.output/'portfolio_M5_M15_H1'
    results = {}
    for tf, upper in PORTFOLIO_MODES:
        engine_cfg = Config(**{**asdict(cfg), 'timeframe': tf, 'higher_timeframe': upper,
                              'use_mtf': True, 'initial_equity': cfg.initial_equity/3})
        end_times = pd.DatetimeIndex(frames[tf].index)+pd.Timedelta(minutes=timeframe_minutes(tf))
        data = frames[tf].loc[end_times <= window_end].copy()
        if len(data) < 2:
            raise ValueError(f'{tf} için yeterli ortak veri yok.')
        upper_ends = pd.DatetimeIndex(frames[upper].index)+pd.Timedelta(minutes=timeframe_minutes(upper))
        higher = frames[upper].loc[upper_ends <= window_end].copy()
        features = apply_mtf_filter(build_features(data, engine_cfg), higher, engine_cfg)
        start_at = int(data.index.searchsorted(window_start))
        trades, equity, summary = backtest(features, engine_cfg, start_at)
        name = f'{tf}-{upper}'
        results[name] = dict(config=engine_cfg, features=features, data=data,
                             higher=higher, trades=trades, equity=equity, summary=summary)
    trades, combined, summary = combine_portfolio(results, cfg, window_start, window_end)
    output.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        folder = output/name
        folder.mkdir(parents=True, exist_ok=True)
        result['trades'].to_csv(folder/'trades.csv', index=False)
        result['equity'].to_csv(folder/'equity.csv', index_label='bar_open_time')
        result['features'].to_csv(folder/'signals.csv', index_label='time')
        result['data'].to_csv(folder/'input_bars.csv', index_label='time')
        result['higher'].to_csv(folder/'higher_timeframe_bars.csv', index_label='time')
        engine_report = {'config': asdict(result['config']), 'metadata': metadata, 'summary': result['summary']}
        (folder/'report.json').write_text(json.dumps(engine_report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        (folder/'rapor.txt').write_text(format_report(result['summary'], result['config'], str(metadata['currency'])), encoding='utf-8')
    trades.to_csv(output/'all_trades.csv', index=False)
    combined.to_csv(output/'portfolio_equity.csv', index_label='close_time')
    engine_summary = [{'engine': name, **{k: v for k, v in result['summary'].items() if not isinstance(v, dict)}} for name, result in results.items()]
    pd.DataFrame(engine_summary).to_csv(output/'engine_summary.csv', index=False)
    report = {'portfolio_config': asdict(cfg), 'metadata': metadata, 'summary': summary,
              'engines': {name: result['summary'] for name, result in results.items()},
              'assumptions': ['Equal capital allocation, independent engines; no capital transfers.',
                  'Risk percentage applies to each allocated engine equity, not full account equity.',
                  'Three simultaneous positions possible; no margin or account stop-out model.',
                  'Aggregate curve uses only available bar CLOSE marks and forward fill.',
                  'OHLC paths and costs retain the existing approximations.']}
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    readable = portfolio_report(results, summary, cfg, str(metadata['currency']))
    if args.compare:
        baseline_results = {}
        for name, result in results.items():
            core_cfg = Config(**{**asdict(result['config']), 'use_mtf': False})
            core_trades, core_equity, core_summary = backtest(build_features(result['data'], core_cfg), core_cfg,
                int(result['data'].index.searchsorted(window_start)))
            baseline_results[name] = dict(config=core_cfg, trades=core_trades, equity=core_equity, summary=core_summary)
        _, _, core_total = combine_portfolio(baseline_results, cfg, window_start, window_end)
        pd.DataFrame([{'version': 'MTF_OFF', **{k:v for k,v in core_total.items() if not isinstance(v,dict)}},
                      {'version': 'MTF_ON', **{k:v for k,v in summary.items() if not isinstance(v,dict)}}]).to_csv(output/'comparison.csv', index=False)
        readable += f"\n\nMTF KARŞILAŞTIRMASI (diğer kurallar aynı)\nMTF kapalı net sonuç: {display_number(core_total['net_pnl'])}\nMTF açık net sonuç  : {display_number(summary['net_pnl'])}\n"
    readable += f'\n\nSonuç klasörü: {output.resolve()}\n'
    (output/'rapor.txt').write_text(readable, encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else readable)


def main():
    reconfigure_stdout = getattr(sys.stdout, 'reconfigure', None)
    if callable(reconfigure_stdout):
        reconfigure_stdout(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Closed BID OHLC bars; optional spread_price in price units.')
    parser.add_argument('--config', type=Path, help='JSON overrides for Config fields.')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'backtest_results')
    parser.add_argument('--compare', action='store_true', help='Compare core and added filters; descriptive, not optimization.')
    parser.add_argument('--test-start', help='UTC evaluation start, e.g. 2025-01-01. Earlier bars only supply indicator/history warmup.')
    parser.add_argument('--json', action='store_true', help='Print the raw JSON summary instead of the Turkish report.')
    parser.add_argument('--mtf', action='store_true', help='Enable closed higher-timeframe MA/AP trend confirmation.')
    parser.add_argument('--mtf-mode', type=str.upper, choices=['M5-H1', 'M15-H4', 'H1-D1'],
                        help='Single engine preset: M5-H1, M15-H4 or H1-D1.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--portfolio', action='store_true', help='Run all three independent engines (default).')
    group.add_argument('--single', action='store_true', help='Run one configured engine instead of default portfolio.')
    parser.add_argument('--data-dir', type=Path, help='Portfolio CSV folder containing M5,M15,H1,H4,D1.csv.')
    parser.add_argument('--higher-timeframe', help='MTF confirmation timeframe, default D1.')
    parser.add_argument('--higher-csv', type=Path, help='Closed higher-timeframe BID OHLC bars for CSV MTF mode.')
    args = parser.parse_args()
    cfg = Config(**(json.loads(args.config.read_text(encoding='utf-8-sig')) if args.config else {}))
    portfolio = args.portfolio or args.data_dir is not None or (not args.single and not args.mtf_mode and not args.csv and not args.mtf and not args.higher_timeframe)
    if portfolio:
        if args.csv or args.higher_csv or args.mtf_mode or args.higher_timeframe or args.single:
            parser.error('Portföy modu için --data-dir veya MT5 kullanın; tek motor seçeneklerini kaldırın.')
        run_portfolio(cfg, args)
        return
    overrides = {}
    if args.mtf:
        overrides['use_mtf'] = True
    if args.higher_timeframe:
        overrides['higher_timeframe'] = args.higher_timeframe.upper()
    if args.mtf_mode:
        if args.higher_timeframe:
            parser.error('--mtf-mode ile --higher-timeframe birlikte kullanılamaz.')
        base_tf, upper_tf = args.mtf_mode.split('-')
        overrides.update({'timeframe': base_tf, 'higher_timeframe': upper_tf, 'use_mtf': True})
    cfg = Config(**{**asdict(cfg), **overrides})
    validate_config(cfg)
    if cfg.use_mtf:
        args.output = args.output / f'mtf_{cfg.timeframe}_{cfg.higher_timeframe}'
    metadata = {'source': 'CSV', 'currency': 'configured account currency'}
    if args.csv:
        raw = pd.read_csv(args.csv)
    else:
        raw, metadata = fetch_mt5(cfg)
        cfg = Config(**{**asdict(cfg), **metadata['contract_config']})
    data = normalize_data(raw, cfg)
    higher = None
    if cfg.use_mtf:
        if args.csv:
            if not args.higher_csv:
                parser.error('CSV MTF için --higher-csv ile üst zaman dilimi verisini belirtin.')
            higher = normalize_data(pd.read_csv(args.higher_csv), cfg)
        else:
            higher = fetch_mt5_higher(cfg, data)
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output/'input_bars.csv', index_label='time')
    features = build_features(data, cfg)
    if higher is not None:
        higher.to_csv(args.output/'higher_timeframe_bars.csv', index_label='time')
        features = apply_mtf_filter(features, higher, cfg)
    start_at = int(data.index.searchsorted(pd.to_datetime(args.test_start, utc=True))) if args.test_start else 0
    trades, equity, summary = backtest(features, cfg, start_at)
    trades.to_csv(args.output/'trades.csv', index=False)
    equity.to_csv(args.output/'equity.csv')
    features.to_csv(args.output/'signals.csv', index_label='time')
    report = {'config': asdict(cfg), 'metadata': metadata, 'summary': summary, 'test_start': args.test_start,
              'assumptions': ['Closed input bars, BID OHLC; short exits use bar spread.',
                  'Lower closing-equity outcome of two OHLC paths; intrabar ordering is approximate.',
                  'Constant commission, slippage, daily carry and PnL conversion; margin is not simulated.',
                  'Drawdown is sampled at bar close; exit timestamps identify bars, not exact fills.',
                  'MTF state becomes available at HTF open plus its duration; only backward time joins are used.',
                  'No parameter search or out-of-sample profitability claim.']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    readable_report = format_report(summary, cfg, str(metadata['currency']))
    if args.compare:
        variants = {'core': {}, 'trend': {'use_trend_filter': True},
                    'adx': {'use_adx_filter': True}, 'volume': {'use_volume_filter': True},
                    'pullback': {'use_pullback': True}, 'without_partial': {'partial_fraction': 0.0}}
        comparisons = []
        base = {**asdict(cfg), 'use_trend_filter': False, 'use_adx_filter': False,
                'use_volume_filter': False, 'use_pullback': False, 'use_mtf': False}
        if cfg.use_mtf:
            variants['mtf'] = {'use_mtf': True}
        for name, override in variants.items():
            variant = Config(**{**base, **override})
            variant_features = build_features(data, variant)
            if variant.use_mtf and higher is not None:
                variant_features = apply_mtf_filter(variant_features, higher, variant)
            _, _, result = backtest(variant_features, variant, start_at)
            comparisons.append({'variant': name, **{k: v for k, v in result.items() if not isinstance(v, dict)}})
        pd.DataFrame(comparisons).to_csv(args.output/'comparison.csv', index=False)
        readable_report += '\n' + format_comparison(comparisons)
    readable_report += f'\n\nSonuç klasörü: {args.output.resolve()}\n'
    (args.output/'rapor.txt').write_text(readable_report, encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False) if args.json else readable_report)


if __name__ == '__main__':
    main()
