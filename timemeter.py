import datetime
import time

_inf: float = float("inf")


class TimeMeter:

    def __init__(self):
        self.start_time: float = 0
        self.present_time: float = 0
        self.elapsed: float = 0
        self.estimation: float = _inf
        self.set_start_time()

    def set_start_time(self):
        self.start_time = time.perf_counter()

    def snap_present_time(self, *, progress: float = 0):
        self.present_time = time.perf_counter()
        self.elapsed = self.present_time - self.start_time
        if progress <= 0 or progress > 1:
            self.estimation = _inf
        else:
            self.estimation = (self.present_time - self.start_time) / progress * (1 - progress)

    @staticmethod
    def format_total_seconds(total_seconds: float, is_detail: bool = False):

        if total_seconds == _inf:
            return "不明"

        td = datetime.timedelta(seconds=total_seconds)
        minutes, seconds = divmod(td.seconds, 60)
        hours, minutes = divmod(minutes, 60)

        result: str = "約 "
        if td.days >= 1:
            result += str(td.days) + " 日 " + str(hours) + " 時間"
        elif hours >= 1:
            result += str(hours) + " 時間 " + str(minutes) + " 分"
        elif minutes >= 1:
            result += str(minutes) + " 分 " + str(seconds) + " 秒"
        else:
            result += str(seconds) + " 秒"

        if is_detail:
            return "{} ({:.3f} 秒)".format(result, total_seconds)
            # 約 1 分 23 秒 (83.456 秒)
        else:
            return result
