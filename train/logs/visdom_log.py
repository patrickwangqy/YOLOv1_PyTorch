import visdom
import torch


class VisdomLineLog(object):
    def __init__(self, vis: visdom.Visdom, name: str):
        self.vis = vis
        self.name = name
        self.log_index = 0

    def append(self, value):
        self.vis.line(X=torch.FloatTensor([self.log_index]),
                      Y=torch.FloatTensor([value]),
                      win=self.name,
                      update="append" if self.log_index > 1 else None,
                      opts={"title": self.name})
        self.log_index += 1


class VisdomLog(object):
    def __init__(self, name: str):
        self.name = name
        self.vis = visdom.Visdom(env=self.name)
        self.log_dict = {}

    def line(self, name: str, value):
        logger = self.log_dict.setdefault(name, VisdomLineLog(self.vis, name))
        logger.append(value)
