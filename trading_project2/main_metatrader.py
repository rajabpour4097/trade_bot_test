import MetaTrader5 as mt5
from datetime import datetime
from fibo_calculate import fibonacci_retracement
import numpy as np
import pandas as pd
from time import sleep
from colorama import init, Fore
from get_legs import get_legs
from mt5_connector import MT5Connector
from swing import get_swing_points
from utils import BotState
from save_file import log
from metatrader5_config import MT5_CONFIG, TRADING_CONFIG, DYNAMIC_RISK_CONFIG, ML_FILTER_CONFIG
from email_notifier import send_trade_email_async
from analytics.hooks import log_signal, log_position_event

# Ensure project root is on sys.path so sibling package imports work
import os, sys
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# اختیاری: فیلتر ML برای تایید معامله
_ML_IMPORT_ERROR = None
try:
    from gpt5_backtest.src.ml_filter import load_model
    from gpt5_backtest.src.ml_features import extract_features
    _ml_model = None
    _ml_features = None
except Exception as _e:
    _ML_IMPORT_ERROR = _e
    # جلوگیری از NameError و غیرفعال‌سازی فیلتر ML
    load_model = None  # type: ignore
    extract_features = None  # type: ignore
    try:
        ML_FILTER_CONFIG['enable'] = False
    except Exception:
        pass
    _ml_model = None
    _ml_features = None


def main():
    # راه‌اندازی MT5 و colorama
    init(autoreset=True)
    mt5_conn = MT5Connector()

    if not mt5_conn.initialize():
        print("❌ Failed to connect to MT5")
        return

    # Initial state با تنظیمات - مطابق main_saver_copy2.py
    state = BotState()
    state.reset()

    start_index = 0
    win_ratio = MT5_CONFIG['win_ratio']
    risk_pct = MT5_CONFIG.get('risk_pct', 0.01)
    sell_only_mode = TRADING_CONFIG.get('sell_only', False)
    threshold = TRADING_CONFIG['threshold']
    window_size = TRADING_CONFIG['window_size']
    min_swing_size = TRADING_CONFIG['min_swing_size']

    i = 1
    f = 1
    position_open = False
    last_swing_type = None
    fib_index = None
    fib0_point = None
    last_leg1_value = None
    end_price = None
    start_price = None

    print(f"🚀 MT5 Trading Bot Started...")
    print(f"📊 Config: Symbol={MT5_CONFIG['symbol']}, Lot={MT5_CONFIG['lot_size']}, Win Ratio={win_ratio}")
    print(f"⏰ Trading Hours (Iran): {MT5_CONFIG['trading_hours']['start']} - {MT5_CONFIG['trading_hours']['end']}")
    print(f"🇮🇷 Current Iran Time: {mt5_conn.get_iran_time().strftime('%Y-%m-%d %H:%M:%S')}")

    # در ابتدای main loop بعد از initialize
    print("🔍 Checking symbol properties...")
    mt5_conn.check_symbol_properties()
    print("🔍 Testing broker filling modes...")
    mt5_conn.test_filling_modes()
    mt5_conn.check_trading_limits()
    print("🔍 Checking account permissions...")
    mt5_conn.check_account_trading_permissions()
    print("🔍 Checking market state...")
    mt5_conn.check_market_state()
    print("-" * 50)

    # بارگذاری مدل ML در صورت فعال بودن
    if ML_FILTER_CONFIG.get('enable'):
        if load_model is None:
            msg = f"ML filter disabled (import error: {_ML_IMPORT_ERROR})" if _ML_IMPORT_ERROR else "ML filter disabled (import missing)"
            log(msg, color='yellow')
        else:
            try:
                _model_path = ML_FILTER_CONFIG.get('model_path')
                if not _model_path:
                    raise ValueError('model_path not set in ML_FILTER_CONFIG')
                # تبدیل مسیر نسبی به مطلق نسبت به ریشه پروژه
                _abs_model_path = _model_path if os.path.isabs(_model_path) else os.path.join(_PROJECT_ROOT, _model_path)
                if not os.path.exists(_abs_model_path):
                    raise FileNotFoundError(f'model file not found: {_abs_model_path}')
                _ml_model, _ml_features = load_model(_abs_model_path)
                log('ML filter loaded', color='cyan')
            except Exception as _e:
                log(f'ML filter load failed: {_e}', color='red')
                _ml_model = None
                _ml_features = None

    # اضافه کردن متغیر برای ذخیره آخرین داده
    last_data_time = None
    wait_count = 0
    max_wait_cycles = 120  # پس از 60 ثانیه (120 * 0.5) اجبار به پردازش

    # بعد از تعریف متغیرها در main()
    def reset_state_and_window():
        nonlocal start_index
        state.reset()
        start_index = max(0, len(cache_data) - window_size)
        log(f'Reset state -> new start_index={start_index} (slice len={len(cache_data.iloc[start_index:])})', color='magenta')

    # حالت‌های مدیریت پوزیشن
    position_states = {}  # ticket -> {'entry':..., 'risk':..., 'direction':..., 'done_stages':set(), 'base_tp_R':float, 'commission_locked':False}

    def _digits():
        info = mt5.symbol_info(MT5_CONFIG['symbol'])
        return info.digits if info else 5

    def _round(p):
        return float(f"{p:.{_digits()}f}")

    def register_position(pos):
        # محاسبه R (ریسک اولیه)
        risk = abs(pos.price_open - pos.sl) if pos.sl else None
        if not risk or risk == 0:
            return
        position_states[pos.ticket] = {
            'entry': pos.price_open,
            'risk': risk,
            'direction': 'buy' if pos.type == mt5.POSITION_TYPE_BUY else 'sell',
            'done_stages': set(),
            'base_tp_R': DYNAMIC_RISK_CONFIG.get('base_tp_R', 1.2),
            'commission_locked': False
        }
        # رویداد ثبت پوزیشن
        try:
            log_position_event(
                symbol=MT5_CONFIG['symbol'],
                ticket=pos.ticket,
                event='open',
                direction=position_states[pos.ticket]['direction'],
                entry=pos.price_open,
                current_price=pos.price_open,
                sl=pos.sl,
                tp=pos.tp,
                profit_R=0.0,
                stage=0,
                risk_abs=risk,
                locked_R=None,
                volume=pos.volume,
                note='position registered'
            )
        except Exception:
            pass

    def manage_open_positions():
        if not DYNAMIC_RISK_CONFIG.get('enable'):
            return
        positions = mt5_conn.get_positions()
        if not positions:
            return
        tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
        if not tick:
            return
        stages_cfg = DYNAMIC_RISK_CONFIG.get('stages', [])
        commission_per_lot = DYNAMIC_RISK_CONFIG.get('commission_per_lot', 0.0)
        for pos in positions:
            if pos.ticket not in position_states:
                register_position(pos)
            st = position_states.get(pos.ticket)
            if not st:
                continue
            entry = st['entry']
            risk = st['risk']
            direction = st['direction']
            cur_price = tick.bid if direction == 'buy' else tick.ask
            # profit in price
            if direction == 'buy':
                price_profit = cur_price - entry
            else:
                price_profit = entry - cur_price
            profit_R = price_profit / risk if risk else 0.0
            modified_any = False

            # محاسبه ارزش پولی 1R تقریبی (بدون اسپرد) برای تبدیل کامیشن به R:
            # risk_abs_price = risk (فاصله قیمتی) * volume * contract ارزش واقعی - ساده‌سازی: فقط نسبت بر اساس فاصله قیمتی.
            # برای دقت بیشتر باید tick_value استفاده شود؛ اینجا ساده نگه می‌داریم.
            # کمیسیون بر اساس حجم پوزیشن جاری:
            commission_total = commission_per_lot * pos.volume if commission_per_lot else 0.0

            # عبور از مراحل
            for stage_cfg in stages_cfg:
                sid = stage_cfg.get('id')
                if sid in st['done_stages']:
                    continue
                new_sl = None
                new_tp = None
                event_name = None
                locked_R = None

                # Stage type commission
                if stage_cfg.get('type') == 'commission' and commission_total > 0:
                    # نیاز به تخمین سود دلاری جاری: تقریباً profit_R * (risk_value)؛ اینجا ساده: اگر profit_R * risk >= commission_value/point_value => پیچیده.
                    # راه ساده: وقتی profit_R * 1R_cash >= commission_total.
                    # ما 1R_cash را تخمینی با (risk * مخصوص ارزش هر پوینت * حجم) باید داشته باشیم؛ چون نداریم، این مرحله را موقتاً بر اساس profit_R >= commission_total / (risk * 100000) (تقریبی) نمی‌کنیم.
                    # ساده‌تر: اگر سود قیمتی * حجم >= کمیسیون (بسیار ساده و تقریبی)
                    if price_profit * pos.volume >= commission_total:
                        # قفل کردن SL روی نقطه ورود + (commission_locked نقطه‌ای)؟ طبق تعریف: «SL روی نقطه پوشش کمیسیون» = entry + offset معادل سود کمیسیون.
                        # offset تقریبی: کمیسیون / (volume)
                        price_offset = commission_total / pos.volume if pos.volume else 0.0
                        if direction == 'buy':
                            new_sl = entry + price_offset
                        else:
                            new_sl = entry - price_offset
                        new_tp = pos.tp  # بدون تغییر
                        event_name = 'commission_cover'
                        locked_R = None
                else:
                    # R-based stage
                    trigger_R = stage_cfg.get('trigger_R')
                    if trigger_R is not None and profit_R >= trigger_R:
                        sl_lock_R = stage_cfg.get('sl_lock_R', trigger_R)
                        tp_R = stage_cfg.get('tp_R')
                        # SL placement
                        if direction == 'buy':
                            new_sl = entry + sl_lock_R * risk
                            if tp_R:
                                new_tp = entry + tp_R * risk
                        else:
                            new_sl = entry - sl_lock_R * risk
                            if tp_R:
                                new_tp = entry - tp_R * risk
                        event_name = sid
                        locked_R = sl_lock_R

                if new_sl is not None:
                    # Round
                    new_sl_r = _round(new_sl)
                    new_tp_r = _round(new_tp) if new_tp is not None else pos.tp
                    # Apply only if improves
                    apply = False
                    if direction == 'buy' and new_sl_r > pos.sl:
                        apply = True
                    if direction == 'sell' and new_sl_r < pos.sl:
                        apply = True
                    if apply:
                        res = mt5_conn.modify_sl_tp(pos.ticket, new_sl=new_sl_r, new_tp=new_tp_r)
                        if res and getattr(res, 'retcode', None) == 10009:
                            st['done_stages'].add(sid)
                            modified_any = True
                            log(f'⚙️ Stage {sid} applied ticket={pos.ticket} SL->{new_sl_r} TP->{new_tp_r}', color='cyan')
                            try:
                                log_position_event(
                                    symbol=MT5_CONFIG['symbol'],
                                    ticket=pos.ticket,
                                    event=event_name or sid,
                                    direction=direction,
                                    entry=entry,
                                    current_price=cur_price,
                                    sl=new_sl_r,
                                    tp=new_tp_r,
                                    profit_R=profit_R,
                                    stage=None,
                                    risk_abs=risk,
                                    locked_R=locked_R,
                                    volume=pos.volume,
                                    note=f'stage {sid} trigger'
                                )
                            except Exception:
                                pass
            if modified_any:
                position_states[pos.ticket] = st

    while True:
        try:
            # بررسی ساعات معاملاتی
            can_trade, trade_message = mt5_conn.can_trade()
            
            if not can_trade:
                log(f"⏰ {trade_message}", color='yellow', save_to_file=False)
                sleep(60)
                continue
            
            # دریافت داده از MT5
            cache_data = mt5_conn.get_historical_data(count=window_size * 2)
            
            if cache_data is None:
                log("❌ Failed to get data from MT5", color='red')
                sleep(5)
                continue
                
            cache_data['status'] = np.where(cache_data['open'] > cache_data['close'], 'bearish', 'bullish')
            
            # بررسی تغییر داده - مشابه main_saver_copy2.py
            current_time = cache_data.index[-1]
            if last_data_time is None:
                log(f"🔄 First run - processing data from {current_time}", color='cyan')
                last_data_time = current_time
                process_data = True
                wait_count = 0
            elif current_time != last_data_time:
                log(f"📊 New data received: {current_time} (previous: {last_data_time})", color='cyan')
                last_data_time = current_time
                process_data = True
                wait_count = 0
            else:
                wait_count += 1
                if wait_count % 20 == 0:  # هر 10 ثانیه یک بار لاگ
                    log(f"⏳ Waiting for new data... Current: {current_time} (wait cycles: {wait_count})", color='yellow')
                
                # اگر خیلی زیاد انتظار کشیدیم، اجبار به پردازش (در صورت تست)
                if wait_count >= max_wait_cycles:
                    log(f"⚠️ Force processing after {wait_count} cycles without new data", color='magenta')
                    process_data = True
                    wait_count = 0
                else:
                    process_data = False
            
            if process_data:
                log((' ' * 80 + '\n') * 3)
                log(f'Log number {i}:', color='lightred_ex')
                log(f' ' * 80)
                i += 1
                
                legs = get_legs(cache_data.iloc[start_index:])
                log(f'First len legs: {len(legs)}', color='green')
                log(f' ' * 80)

                if len(legs) > 2:
                    legs = legs[-3:]
                    swing_type, is_swing = get_swing_points(data=cache_data, legs=legs)

                    if is_swing == False and state.fib_levels is None:
                        log(f'No swing or fib levels and legs>2', color='blue')
                        log(f"{cache_data.loc[legs[0]['start']].name} {cache_data.loc[legs[0]['end']].name} "
                            f"{cache_data.loc[legs[1]['start']].name} {cache_data.loc[legs[1]['end']].name} "
                            f"{cache_data.loc[legs[2]['start']].name} {cache_data.loc[legs[2]['end']].name}", color='yellow')

                    if is_swing or state.fib_levels:
                        log(f'1- is_swing or fib_levels is not None code:411112', color='blue')
                        log(f"{swing_type} | {cache_data.loc[legs[0]['start']].name} {cache_data.loc[legs[0]['end']].name} "
                            f"{cache_data.loc[legs[1]['start']].name} {cache_data.loc[legs[1]['end']].name} "
                            f"{cache_data.loc[legs[2]['start']].name} {cache_data.loc[legs[2]['end']].name}", color='yellow')

                        log(f' ' * 80)
                        
                        # فاز 1: تشخیص اولیه swing
                        if is_swing and state.fib_levels is None:
                            log(f'is_swing and fib_levels is None code:4113312', color='yellow')
                            last_swing_type = swing_type
                            if swing_type == 'bullish':
                                if cache_data.iloc[-1]['close'] >= legs[0]['end_value']:
                                    # دقیقا مثل main_saver_copy2.py
                                    start_price = cache_data.iloc[-1]['high']
                                    end_price = legs[1]['end_value']
                                    if cache_data.iloc[-1]['high'] >= legs[1]['end_value']:
                                        log(f'The {f} of fib_levels value code:4116455 {cache_data.iloc[-1].name}', color='green')
                                        state.fib_levels = fibonacci_retracement(end_price=end_price, start_price=start_price)
                                        fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                        fib_index = cache_data.iloc[-1].name
                                        last_leg1_value = cache_data.index.tolist().index(legs[1]['end'])
                                        legs = legs[-2:]
                                        f += 1
                                    elif state.fib_levels and cache_data.iloc[-1]['low'] < state.fib_levels['1.0']:
                                        reset_state_and_window()
                                        legs = []
                                    if state.fib_levels:
                                        log(f'fib_levels: {state.fib_levels}', color='yellow')
                                        log(f'fib_index: {fib_index}', color='yellow')
                            elif swing_type == 'bearish':
                                if cache_data.iloc[-1]['close'] <= legs[0]['end_value']:
                                    start_price = cache_data.iloc[-1]['low']
                                    end_price = legs[1]['end_value']
                                    if cache_data.iloc[-1]['low'] <= legs[1]['end_value']:
                                        log(f'The {f} of fib_levels value code:4126455 {cache_data.iloc[-1].name}', color='green')
                                        state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
                                        fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                        fib_index = cache_data.iloc[-1].name
                                        last_leg1_value = cache_data.index.tolist().index(legs[1]['end'])
                                        legs = legs[-2:]
                                        f += 1
                                    elif state.fib_levels and cache_data.iloc[-1]['high'] > state.fib_levels['1.0']:
                                        reset_state_and_window()
                                        legs = []
                                    if state.fib_levels:
                                        log(f'fib_levels: {state.fib_levels}', color='yellow')
                                        log(f'fib_index: {fib_index}', color='yellow')

                        # فاز 2: به‌روزرسانی در swing مشابه - تنها در صورت یکسان بودن جهت
                        elif is_swing and state.fib_levels and last_swing_type == swing_type: 
                            log(f'is_swing and state.fib_levels and last_swing_type == swing_type code:4213312', color='yellow')
                            if swing_type == 'bullish': 
                                if cache_data.iloc[-1]['high'] >= legs[1]['end_value']:
                                    log(f'The {f} of fib_levels value update code:9916455 {cache_data.iloc[-1].name}', color='green')
                                    start_price = cache_data.iloc[-1]['high']
                                    end_price = legs[1]['end_value']
                                    state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
                                    fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                    fib_index = cache_data.iloc[-1].name
                                    last_leg1_value = cache_data.index.tolist().index(legs[1]['end'])
                                    legs = legs[-2:]
                                    f += 1
                                elif cache_data.iloc[-1]['low'] <= state.fib_levels['0.705']:
                                    if state.last_touched_705_point_up is None:
                                        log(f'first touch 705 point code:7318455', color='green')
                                        state.last_touched_705_point_up = cache_data.iloc[-1]
                                    elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_up['status']:
                                        log(f'Second touch 705 point code:7218455 {cache_data.iloc[-1].name}', color='green')
                                        state.true_position = True      
                                elif state.fib_levels and cache_data.iloc[-1]['low'] < state.fib_levels['1.0']:
                                    reset_state_and_window()
                                    legs = []
                            elif swing_type == 'bearish':
                                if cache_data.iloc[-1]['low'] <= legs[1]['end_value']:
                                    log(f'The {f} of fib_levels value update code:9916455 {cache_data.iloc[-1].name}', color='green')
                                    start_price = cache_data.iloc[-1]['low']
                                    end_price = legs[1]['end_value']
                                    state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
                                    fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                    fib_index = cache_data.iloc[-1].name
                                    last_leg1_value = cache_data.index.tolist().index(legs[1]['end'])
                                    legs = legs[-2:]
                                    f += 1
                                elif cache_data.iloc[-1]['high'] >= state.fib_levels['0.705']:
                                    if state.last_touched_705_point_down is None:
                                        log(f'first touch 705 point code:6328455', color='red')
                                        state.last_touched_705_point_down = cache_data.iloc[-1]
                                    elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_down['status']:
                                        log(f'Second touch 705 point code:6228455 {cache_data.iloc[-1].name}', color='green')
                                        state.true_position = True
                                elif state.fib_levels and cache_data.iloc[-1]['high'] > state.fib_levels['1.0']:
                                    reset_state_and_window()
                                    legs = []

                        # فاز جدید: مدیریت swing معکوس بدون عبور از fib 1.0
                        elif is_swing and state.fib_levels and last_swing_type != swing_type:
                            log(f'is_swing with opposite direction - checking fib 1.0 violation', color='orange')
                            if last_swing_type == 'bullish' and swing_type == 'bearish':
                                if cache_data.iloc[-1]['low'] < state.fib_levels['1.0']:
                                    log(f'Bearish swing violated fib 1.0 - resetting', color='red')
                                    state.reset()
                                    legs = legs[-3:]
                                    start_index = cache_data.index.tolist().index(legs[0]['start'])
                                else:
                                    log(f'Bearish swing within fib range - ignoring', color='yellow')
                            elif last_swing_type == 'bearish' and swing_type == 'bullish':
                                if cache_data.iloc[-1]['high'] > state.fib_levels['1.0']:
                                    log(f'Bullish swing violated fib 1.0 - resetting', color='red')
                                    state.reset()
                                    legs = legs[-3:]
                                    start_index = cache_data.index.tolist().index(legs[0]['start'])
                                else:
                                    log(f'Bullish swing within fib range - ignoring', color='yellow')

                        elif is_swing == False and state.fib_levels:
                            if last_swing_type == 'bullish':
                                start_price = cache_data.iloc[-1]['high']
                                end_price = legs[1]['end_value']
                                if state.fib_levels['0.0'] < cache_data.iloc[-1]['high']:
                                    log(f'update fib_levels value code:7117455 {cache_data.iloc[-1].name}', color='green')
                                    state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
                                    fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                    fib_index = cache_data.iloc[-1].name
                                elif cache_data.iloc[-1]['low'] <= state.fib_levels['0.705']:
                                    if state.last_touched_705_point_up is None:
                                        log(f'first touch 705 point code:7318455', color='green')
                                        state.last_touched_705_point_up = cache_data.iloc[-1]
                                    elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_up['status']:
                                        log(f'Second touch 705 point code:7218455 {cache_data.iloc[-1].name}', color='green')
                                        state.true_position = True      
                                elif state.fib_levels and cache_data.iloc[-1]['low'] < state.fib_levels['1.0']:
                                    reset_state_and_window()
                                    legs = []
                            if last_swing_type == 'bearish':
                                start_price = cache_data.iloc[-1]['low']
                                end_price = legs[1]['end_value']
                                if state.fib_levels['0.0'] > cache_data.iloc[-1]['low']:
                                    log(f'update fib_levels value code:6127455 {cache_data.iloc[-1].name}', color='green')
                                    state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
                                    fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                    fib_index = cache_data.iloc[-1].name
                                elif cache_data.iloc[-1]['high'] >= state.fib_levels['0.705']:
                                    if state.last_touched_705_point_down is None:
                                        log(f'first touch 705 point code:6328455', color='red')
                                        state.last_touched_705_point_down = cache_data.iloc[-1]
                                    elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_down['status']:
                                        log(f'Second touch 705 point code:6228455 {cache_data.iloc[-1].name}', color='green')
                                        state.true_position = True
                                elif state.fib_levels and cache_data.iloc[-1]['high'] > state.fib_levels['1.0']:
                                    reset_state_and_window()
                                    legs = []

                elif len(legs) < 3:
                    if state.fib_levels:
                        if last_swing_type == 'bullish' or swing_type == 'bullish':
                            if state.fib_levels['0.0'] < cache_data.iloc[-1]['high']:
                                log(f'update fib_levels value code:5117455 {cache_data.iloc[-1].name}', color='green')
                                start_price = cache_data.iloc[-1]['high']
                                state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name
                            elif cache_data.iloc[-1]['low'] <= state.fib_levels['0.705']:
                                if state.last_touched_705_point_up is None:
                                    log(f'first touch 705 point', color='green')
                                    state.last_touched_705_point_up = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_up['status']:
                                    log(f'Second touch 705 point code:4118455 {cache_data.iloc[-1].name}', color='green')
                                    state.true_position = True                               
                        if last_swing_type == 'bearish' or swing_type == 'bearish':
                            if state.fib_levels['0.0'] > cache_data.iloc[-1]['low']:
                                log(f'update fib_levels value code:5127455 {cache_data.iloc[-1].name}', color='green')
                                start_price = cache_data.iloc[-1]['low']
                                state.fib_levels = fibonacci_retracement(start_price=start_price, end_price=end_price)
                                fib0_point = cache_data.index.tolist().index(cache_data.iloc[-1].name)
                                fib_index = cache_data.iloc[-1].name
                            elif cache_data.iloc[-1]['high'] >= state.fib_levels['0.705']:
                                if state.last_touched_705_point_down is None:
                                    log(f'first touch 705 point', color='red')
                                    state.last_touched_705_point_down = cache_data.iloc[-1]
                                elif cache_data.iloc[-1]['status'] != state.last_touched_705_point_down['status']:
                                    log(f'Second touch 705 point code:5128455 {cache_data.iloc[-1].name}', color='green')
                                    state.true_position = True
                    if len(legs) == 2:
                        log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}, leg1: {legs[1]["start"]}, {legs[1]["end"]}', color='lightcyan_ex')
                    if len(legs) == 1:
                        log(f'leg0: {legs[0]["start"]}, {legs[0]["end"]}', color='lightcyan_ex')
                
                # بخش معاملات - buy statement (مطابق منطق main_saver_copy2.py)
                if state.true_position and (last_swing_type == 'bullish' or swing_type == 'bullish'):
                    # اگر در حالت SELL-only هستیم، خرید را رد کن
                    if sell_only_mode:
                        log('🚫 BUY skipped (SELL-only mode)', color='red')
                        state.reset(); reset_state_and_window(); legs = []
                        continue
                    # در صورت فعال بودن ML، تایید BUY
                    if ML_FILTER_CONFIG.get('enable') and _ml_model is not None:
                        try:
                            # تشکیل فیچرها از آخرین کندل و فیب و ایمپالس اخیر
                            if len(legs) >= 3:
                                impulse = legs[-3]
                            else:
                                log('🚫 BUY blocked: insufficient legs for ML context', color='red')
                                state.reset(); reset_state_and_window(); legs = []
                                continue
                            fib = state.fib_levels
                            feats = extract_features(cache_data, len(cache_data)-1, 'bullish',
                                                     {'start_value': impulse['start_value'], 'end_value': impulse['end_value']},
                                                     {'0.705': fib['0.705'], '0.9': fib['0.9']},
                                                     ML_FILTER_CONFIG.get('lookback_bars', 140))
                            cols = _ml_features or list(feats.keys())
                            import pandas as _pd
                            _x = _pd.DataFrame([[feats.get(c, 0.0) for c in cols]], columns=cols)
                            try:
                                _proba = float(_ml_model.predict_proba(_x)[0][1])
                            except Exception:
                                _pred = float(_ml_model.predict(_x)[0])
                                _proba = 1.0 if _pred >= 0.5 else 0.0
                            if _proba < ML_FILTER_CONFIG.get('prob_threshold', 0.8):
                                log(f'🚫 BUY blocked by ML filter (p={_proba:.2f})', color='red')
                                state.reset(); reset_state_and_window(); legs = []
                                continue
                            else:
                                log(f'✅ BUY passed ML filter (p={_proba:.2f})', color='green')
                        except Exception as _e:
                            log(f'ML BUY filter error: {_e}', color='red')
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    buy_entry_price = last_tick.ask
                    # لاگ سیگنال (قبل از ارسال سفارش)
                    try:
                        log_signal(
                            symbol=MT5_CONFIG['symbol'],
                            strategy="swing_fib_v1",
                            direction="buy",
                            rr=win_ratio,
                            entry=buy_entry_price,
                            sl=float(state.fib_levels['1.0'] if abs(state.fib_levels['0.9']-buy_entry_price) <= _pip_size_for(MT5_CONFIG['symbol'])*2 else state.fib_levels['0.9']),
                            tp=None,
                            fib=state.fib_levels,
                            confidence=None,
                            features_json=None,
                            note="triggered_by_pullback"
                        )
                    except Exception:
                        pass
                    # دریافت قیمت لحظه‌ای بازار از MT5
                    # current_open_point = cache_data.iloc[-1]['close']
                    log(f'Start long position income {cache_data.iloc[-1].name}', color='blue')
                    log(f'current_open_point (market ask): {buy_entry_price}', color='blue')

                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    two_pips = 2.0 * pip_size
                    min_dist = _min_stop_distance(MT5_CONFIG['symbol'])

                    # معیار درستِ 2 پیپ
                    is_close_to_09 = abs(state.fib_levels['0.9'] - buy_entry_price) <= two_pips

                    candidate_sl = state.fib_levels['1.0'] if is_close_to_09 else state.fib_levels['0.9']

                    min_pip_dist = 2  # حداقل 2 پیپ واقعی
                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    min_abs_dist = max(min_pip_dist * pip_size, min_dist)

                    # گارد جهت
                    if candidate_sl >= buy_entry_price:
                        # برگرداندن به 1.0 اگر 0.9 بالاتر بود
                        candidate_sl = float(state.fib_levels['1.0'])
                    # اطمینان از فاصله
                    if (buy_entry_price - candidate_sl) < min_abs_dist:
                        # اگر فاصله خیلی کم است، یا SL را جابه‌جا کن یا معامله را لغو کن
                        adj = buy_entry_price - min_abs_dist
                        if adj <= 0:
                            log("🚫 Skip BUY: invalid SL distance", color='red')
                            state.reset()
                            reset_state_and_window()
                            continue
                        candidate_sl = float(adj)

                    stop = float(candidate_sl)
                    if stop >= buy_entry_price:
                        log("🚫 Skip BUY: SL still >= entry after adjust", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue

                    stop_distance = abs(buy_entry_price - stop)
                    # بررسی حداقل فاصله SL
                    min_sl_pips = TRADING_CONFIG.get('min_sl_pips', 2.0)
                    pip_size_check = _pip_size_for(MT5_CONFIG['symbol'])
                    if stop_distance < (min_sl_pips * pip_size_check):
                        log(f"🚫 Skip BUY: SL distance {stop_distance:.6f} < {min_sl_pips} pips", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue
                    reward_end = buy_entry_price + (stop_distance * win_ratio)
                    log(f'stop = {stop}', color='green')
                    log(f'reward_end = {reward_end}', color='green')

                    # ارسال سفارش BUY با هر stop و reward
                    result = mt5_conn.open_buy_position(
                        tick=last_tick,
                        sl=stop,
                        tp=reward_end,
                        comment=f"Bullish Swing {swing_type}",
                        risk_pct=risk_pct  # از تنظیمات
                    )
                    # ارسال ایمیل غیرمسدودکننده
                    try:
                        send_trade_email_async(
                            subject=f"TEST ROBOT: NEW BUY ORDER {MT5_CONFIG['symbol']}",
                            body=(
                                f"Time: {datetime.now()}\n"
                                f"Symbol: {MT5_CONFIG['symbol']}\n"
                                f"Type: BUY (Bullish Swing)\n"
                                f"Entry: {buy_entry_price}\n"
                                f"SL: {stop}\n"
                                f"TP: {reward_end}\n"
                            )
                        )
                    except Exception as _e:
                        log(f'Email dispatch failed: {_e}', color='red')

                    if result and getattr(result, 'retcode', None) == 10009:
                        log(f'✅ BUY order executed successfully', color='green')
                        log(f'📊 Ticket={result.order} Price={result.price} Volume={result.volume}', color='cyan')
                        # ارسال ایمیل غیرمسدودکننده
                        try:
                            send_trade_email_async(
                                subject = f"TEST ROBOT: Last order result",
                                body=(
                                    f"Ticket={result.order}\n"
                                    f"Price={result.price}\n"
                                    f"Volume={result.volume}\n"
                                )
                            )
                        except Exception as _e:
                            log(f'Email dispatch failed: {_e}', color='red')
                    else:
                        if result:
                            log(f'❌ BUY failed retcode={result.retcode} comment={result.comment}', color='red')
                        else:
                            log(f'❌ BUY failed (no result object)', color='red')
                    state.reset()

                    reset_state_and_window()
                    legs = []

                # بخش معاملات - sell statement (مطابق منطق main_saver_copy2.py)
                if state.true_position and (last_swing_type == 'bearish' or swing_type == 'bearish'):
                    # در صورت فعال بودن ML، تایید SELL
                    if ML_FILTER_CONFIG.get('enable') and _ml_model is not None:
                        try:
                            if len(legs) >= 3:
                                impulse = legs[-3]
                            else:
                                log('🚫 SELL blocked: insufficient legs for ML context', color='red')
                                state.reset(); reset_state_and_window(); legs = []
                                continue
                            fib = state.fib_levels
                            feats = extract_features(cache_data, len(cache_data)-1, 'bearish',
                                                     {'start_value': impulse['start_value'], 'end_value': impulse['end_value']},
                                                     {'0.705': fib['0.705'], '0.9': fib['0.9']},
                                                     ML_FILTER_CONFIG.get('lookback_bars', 140))
                            cols = _ml_features or list(feats.keys())
                            import pandas as _pd
                            _x = _pd.DataFrame([[feats.get(c, 0.0) for c in cols]], columns=cols)
                            try:
                                _proba = float(_ml_model.predict_proba(_x)[0][1])
                            except Exception:
                                _pred = float(_ml_model.predict(_x)[0])
                                _proba = 1.0 if _pred >= 0.5 else 0.0
                            if _proba < ML_FILTER_CONFIG.get('prob_threshold', 0.8):
                                log(f'🚫 SELL blocked by ML filter (p={_proba:.2f})', color='red')
                                state.reset(); reset_state_and_window(); legs = []
                                continue
                            else:
                                log(f'✅ SELL passed ML filter (p={_proba:.2f})', color='green')
                        except Exception as _e:
                            log(f'ML SELL filter error: {_e}', color='red')
                    last_tick = mt5.symbol_info_tick(MT5_CONFIG['symbol'])
                    sell_entry_price = last_tick.bid
                    try:
                        log_signal(
                            symbol=MT5_CONFIG['symbol'],
                            strategy="swing_fib_v1",
                            direction="sell",
                            rr=win_ratio,
                            entry=sell_entry_price,
                            sl=float(state.fib_levels['1.0'] if abs(state.fib_levels['0.9']-sell_entry_price) <= _pip_size_for(MT5_CONFIG['symbol'])*2 else state.fib_levels['0.9']),
                            tp=None,
                            fib=state.fib_levels,
                            confidence=None,
                            features_json=None,
                            note="triggered_by_pullback"
                        )
                    except Exception:
                        pass
                    log(f'Start short position income {cache_data.iloc[-1].name}', color='red')
                    log(f'current_open_point (market bid): {sell_entry_price}', color='red')

                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    two_pips = 2.0 * pip_size
                    min_dist = _min_stop_distance(MT5_CONFIG['symbol'])

                    is_close_to_09 = abs(state.fib_levels['0.9'] - sell_entry_price) <= two_pips
                    candidate_sl = state.fib_levels['1.0'] if is_close_to_09 else state.fib_levels['0.9']

                    min_pip_dist = 2.0
                    pip_size = _pip_size_for(MT5_CONFIG['symbol'])
                    min_abs_dist = max(min_pip_dist * pip_size, min_dist)

                    if candidate_sl <= sell_entry_price:
                        candidate_sl = float(state.fib_levels['1.0'])
                    if (candidate_sl - sell_entry_price) < min_abs_dist:
                        adj = sell_entry_price + min_abs_dist
                        candidate_sl = float(adj)

                    stop = float(candidate_sl)
                    if stop <= sell_entry_price:
                        log("🚫 Skip SELL: SL still <= entry after adjust", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue

                    stop_distance = abs(sell_entry_price - stop)
                    min_sl_pips = TRADING_CONFIG.get('min_sl_pips', 2.0)
                    pip_size_check = _pip_size_for(MT5_CONFIG['symbol'])
                    if stop_distance < (min_sl_pips * pip_size_check):
                        log(f"🚫 Skip SELL: SL distance {stop_distance:.6f} < {min_sl_pips} pips", color='red')
                        state.reset()
                        reset_state_and_window()
                        continue
                    reward_end = sell_entry_price - (stop_distance * win_ratio)
                    log(f'stop = {stop}', color='red')
                    log(f'reward_end = {reward_end}', color='red')

                    # ارسال سفارش SELL با هر stop و reward
                    result = mt5_conn.open_sell_position(
                        tick=last_tick,
                        sl=stop,
                        tp=reward_end,
                        comment=f"Bearish Swing {swing_type}",
                        risk_pct=risk_pct  # از تنظیمات
                    )
                    
                    # ارسال ایمیل غیرمسدودکننده
                    try:
                        send_trade_email_async(
                            subject=f"TEST ROBOT: NEW SELL ORDER {MT5_CONFIG['symbol']}",
                            body=(
                                f"Time: {datetime.now()}\n"
                                f"Symbol: {MT5_CONFIG['symbol']}\n"
                                f"Type: SELL (Bearish Swing)\n"
                                f"Entry: {sell_entry_price}\n"
                                f"SL: {stop}\n"
                                f"TP: {reward_end}\n"
                            )
                        )
                    except Exception as _e:
                        log(f'Email dispatch failed: {_e}', color='red')
                    
                    if result and getattr(result, 'retcode', None) == 10009:
                        log(f'✅ SELL order executed successfully', color='green')
                        log(f'📊 Ticket={result.order} Price={result.price} Volume={result.volume}', color='cyan')
                        # ارسال ایمیل غیرمسدودکننده
                        try:
                            send_trade_email_async(
                                subject = f"TEST ROBOT: Last order result",
                                body=(
                                    f"Ticket={result.order}\n"
                                    f"Price={result.price}\n"
                                    f"Volume={result.volume}\n"
                                )
                            )
                        except Exception as _e:
                            log(f'Email dispatch failed: {_e}', color='red')
                    else:
                        if result:
                            log(f'❌ SELL failed retcode={result.retcode} comment={result.comment}', color='red')
                        else:
                            log(f'❌ SELL failed (no result object)', color='red')
                    state.reset()

                    reset_state_and_window()
                    legs = []
                
                log(f'cache_data.iloc[-1].name: {cache_data.iloc[-1].name}', color='lightblue_ex')
                log(f'len(legs): {len(legs)} | start_index: {start_index} | {cache_data.iloc[start_index].name}', color='lightred_ex')
                log(f' ' * 80)
                log(f'-'* 80)
                log(f' ' * 80)

                # ذخیره آخرین زمان داده
                # last_data_time = cache_data.index[-1]  # این خط حذف شد چون بالا انجام شد

            # بررسی وضعیت پوزیشن‌های باز
            positions = mt5_conn.get_positions()
            if positions is None or len(positions) == 0:
                if position_open:
                    log("🏁 Position closed", color='yellow')
                    position_open = False

            manage_open_positions()

            sleep(0.5)  # مطابق main_saver_copy2.py

        except KeyboardInterrupt:
            log("🛑 Bot stopped by user", color='yellow')
            mt5_conn.close_all_positions()
            break
        except Exception as e:
            log(f' ' * 80)
            log(f"❌ Error: {e}", color='red')
            sleep(5)

    mt5_conn.shutdown()
    print("🔌 MT5 connection closed")

def _pip_size_for(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0001
    # برای 5/3 رقمی: 1 pip = 10 * point
    return info.point * (10.0 if info.digits in (3, 5) else 1.0)

def _min_stop_distance(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    if not info:
        return 0.0003
    point = info.point
    # حداقل فاصله مجاز بروکر (stops_level) یا 3 پوینت به‌عنوان فfallback
    return max((getattr(info, 'trade_stops_level', 0) or 0) * point, 3 * point)

if __name__ == "__main__":
    main()