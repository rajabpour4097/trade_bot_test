# ساعات مختلف بازار فارکس بر اساس ساعت ایران

# جلسه سیدنی (05:30 - 14:30 ایران)
SYDNEY_HOURS_IRAN = {
    'start': '05:30',
    'end': '14:30'
}

# جلسه توکیو (07:30 - 16:30 ایران)  
TOKYO_HOURS_IRAN = {
    'start': '07:30',
    'end': '16:30'
}

# جلسه لندن (12:30 - 21:30 ایران)
LONDON_HOURS_IRAN = {
    'start': '12:30',
    'end': '21:30'
}

# جلسه نیویورک (17:30 - 02:30 ایران)
NEWYORK_HOURS_IRAN = {
    'start': '17:30',
    'end': '02:30'  # روز بعد
}

# همپوشانی لندن-نیویورک (17:30 - 21:30 ایران) - بهترین زمان
OVERLAP_LONDON_NY_IRAN = {
    'start': '17:30',
    'end': '21:30'
}

# ساعات فعال ایرانی (09:00 - 21:00)
IRAN_ACTIVE_HOURS = {
    'start': '09:00',
    'end': '21:00'
}

# 24 ساعته
FULL_TIME_IRAN = {
    'start': '00:00',
    'end': '23:59'
}

# تنظیمات MT5
MT5_CONFIG = {
    'symbol': 'EURUSD',
    'lot_size': 0.01,
    'win_ratio': 1.2,
    'risk_pct': 0.01,           # 1.0% ریسک برای تست یک‌هفته‌ای
    'magic_number': 234000,
    'deviation': 20,
    'max_spread': 3.0,
    'min_balance': 100,
    'max_daily_trades': 10,
    'trading_hours': OVERLAP_LONDON_NY_IRAN,
}

# تنظیمات استراتژی
TRADING_CONFIG = {
    'threshold': 7,             # سخت‌تر برای کاهش نویز
    'fib_705': 0.705,
    'fib_90': 0.9,
    'window_size': 100,
    'min_swing_size': 4,
    'entry_tolerance': 2.0,
    'lookback_period': 20,
    'min_sl_pips': 2.0,      # حداقل فاصله مجاز SL از ورود (بر حسب پیپ)؛ کمتر از این = عدم ورود
    'sell_only': True,        # برای تست یک‌هفته‌ای فقط SELL
}

# فیلتر ML برای تایید معاملات
ML_FILTER_CONFIG = {
    'enable': True,                              # فعال برای تست یک‌هفته‌ای
    'model_path': 'gpt5_backtest/results/ml/model.joblib',
    'prob_threshold': 0.80,                      # آستانه پذیرش معامله (0..1)
    'lookback_bars': 140,                        # هم‌راستا با داده آموزشی
}

# مدیریت پویا چند مرحله‌ای جدید
# مراحل:
# 1) پوشش کارمزد: وقتی سود شناور برابر یا بیشتر از مجموع کمیسیون شد، SL روی نقطه پوشش کارمزد قرار می‌گیرد (قفل همان مقدار) و TP اصلی (1.2R) حفظ می‌شود.
# 2) 0.5R: SL روی +0.5R قرار می‌گیرد، TP همچنان 1.2R
# 3) 1.0R: SL روی +1.0R، TP به 1.5R افزایش
# 4) 1.5R: SL روی +1.5R، TP به 2.0R افزایش
DYNAMIC_RISK_CONFIG = {
    'enable': True,
    'commission_per_lot': 4.5,          # کمیسیون کل (رفت و برگشت یا فقط رفت؟ طبق بروکر - قابل تنظیم)
    'commission_mode': 'per_lot',       # per_lot (کل)، per_side (نیمی از رفت و برگشت) در صورت نیاز توسعه
    'round_trip': False,                # اگر True و per_side باشد دو برابر می‌کند
    'base_tp_R': 1.2,                   # TP اولیه تنظیم‌شده هنگام ورود (برای مرجع)
    'stages': [
        {  # پوشش کمیسیون
            'id': 'commission_cover',
            'type': 'commission',      # trigger by commission recovered
            'sl_lock': 'commission',   # قفل روی مقدار کمیسیون
            'tp_R': 1.2                # بدون تغییر
        },
        {  # 0.5R
            'id': 'half_R',
            'trigger_R': 0.5,
            'sl_lock_R': 0.5,
            'tp_R': 1.2
        },
        {  # 1.0R
            'id': 'one_R',
            'trigger_R': 1.0,
            'sl_lock_R': 1.0,
            'tp_R': 1.5
        },
        {  # 1.5R
            'id': 'one_half_R',
            'trigger_R': 1.5,
            'sl_lock_R': 1.5,
            'tp_R': 2.0
        }
    ]
}

# تنظیمات لاگ
LOG_CONFIG = {
    'log_level': 'INFO',        # DEBUG, INFO, WARNING, ERROR
    'save_to_file': True,       # ذخیره در فایل
    'max_log_size': 10,         # حداکثر حجم فایل لاگ (MB)
}