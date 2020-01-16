import torch
import timeit
import os
import datetime

from train.logs import VisdomLog as Log


class Train(object):
    def __init__(self, model, optimizer, criterion, epochs, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.logger = Log("yolov1")
        self.checkpoint_dir = "checkpoints"

    def fit(self, trainloader, statistics_steps=1, valloader=None):
        print("train")
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch, trainloader, statistics_steps)
            if valloader:
                self.evaluate(valloader)

    def train_epoch(self, epoch, trainloader, statistics_steps=1):
        self.model.train()
        running_loss = 0.0
        running_class_loss = 0.0
        running_xy_loss = 0.0
        running_wh_loss = 0.0
        running_pos_conf_loss = 0.0
        running_neg_conf_loss = 0.0
        steps_time = 0.0
        for step, data in enumerate(trainloader, 1):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            start_time = timeit.default_timer()
            loss, class_loss, xy_loss, wh_loss, pos_conf_loss, neg_conf_loss = self.train_step(inputs, labels)
            steps_time += timeit.default_timer() - start_time

            loss = loss.item()
            class_loss = class_loss.item()
            xy_loss = xy_loss.item()
            wh_loss = wh_loss.item()
            pos_conf_loss = pos_conf_loss.item()
            neg_conf_loss = neg_conf_loss.item()

            running_loss += loss
            running_class_loss += class_loss
            running_xy_loss += xy_loss
            running_wh_loss += wh_loss
            running_pos_conf_loss += pos_conf_loss
            running_neg_conf_loss += neg_conf_loss
            if step % statistics_steps == 0:
                running_loss /= statistics_steps
                running_class_loss /= statistics_steps
                running_xy_loss /= statistics_steps
                running_wh_loss /= statistics_steps
                running_pos_conf_loss /= statistics_steps
                running_neg_conf_loss /= statistics_steps

                self.logger.line("loss", running_loss)
                self.logger.line("class loss", running_class_loss)
                self.logger.line("xy loss", running_xy_loss)
                self.logger.line("wh loss", running_wh_loss)
                self.logger.line("pos conf loss", running_pos_conf_loss)
                self.logger.line("neg conf loss", running_neg_conf_loss)

                print(f"{datetime.datetime.now()}  "
                      f"epoch:{epoch}  "
                      f"step:{step}  "
                      f"time:{steps_time:.3f}s  ")
                print(f"\t"
                      f"loss: {running_loss:.3f}  "
                      f"class loss: {running_class_loss:.3f}  "
                      f"xy loss: {running_xy_loss:.3f}  "
                      f"wh loss: {running_wh_loss:.3f}  "
                      f"pos conf loss: {running_pos_conf_loss:.3f}  "
                      f"neg conf loss: {running_neg_conf_loss:.3f}  ")

                running_loss = 0.0
                running_class_loss = 0.0
                running_xy_loss = 0.0
                running_wh_loss = 0.0
                running_pos_conf_loss = 0.0
                running_neg_conf_loss = 0.0
                steps_time = 0.0
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"{epoch:04d}.pt"))

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        loss, class_loss, xy_loss, wh_loss, pos_conf_loss, neg_conf_loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss, class_loss, xy_loss, wh_loss, pos_conf_loss, neg_conf_loss

    def evaluate(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'accuracy: {correct / total:.2%}')
