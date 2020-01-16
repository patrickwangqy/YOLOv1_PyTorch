import tensorboardX
import datetime
from typing import Dict


class TBLog(object):
    def __init__(self, name: str, log_dir: str = "runs"):
        self.writer = tensorboardX.SummaryWriter(f"{log_dir}/{name}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.indexes: Dict[str, int] = {}

    def line(self, name: str, value: float):
        self.writer.add_scalar(name, value, self.indexes.setdefault(name, 0))
        self.indexes[name] += 1
