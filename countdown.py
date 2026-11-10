import datetime
import time


def get_countdown_string():
    now = datetime.datetime.now(datetime.timezone.utc)
    tomorrow = now.date() + datetime.timedelta(days=1)
    bar_close_time = datetime.datetime.combine(tomorrow, datetime.time.min, datetime.timezone.utc)
    time_remaining = bar_close_time - now
    total_seconds = int(time_remaining.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


print("Ticking TradingView 1-Day Bar Close Countdown (UTC):")
try:
    while True:
        countdown = get_countdown_string()
        print(f"\rRemaining Time: {countdown}", end="", flush=True)
        time.sleep(1)
except KeyboardInterrupt:
    print("\nCountdown stopped.")
