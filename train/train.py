import torch
import timeit

from train.logs import VisdomLog


class Train(object):
    def __init__(self, model, optimizer, criterion, epochs, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.logger = VisdomLog("yolov1 train")

    def fit(self, trainloader, statistics_steps=1, valloader=None):
        print("train")
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch, trainloader, statistics_steps)
            if valloader:
                self.evaluate(valloader)

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()

        outputs = self.model(inputs)
        class_loss, object_confidence_loss, no_object_confidence_loss, coord_loss = self.criterion(labels, outputs)
        loss = class_loss + 2 * object_confidence_loss + 0.5 * no_object_confidence_loss + 5 * coord_loss
        loss.backward()
        self.optimizer.step()

        return class_loss, object_confidence_loss, no_object_confidence_loss, coord_loss

    def train_epoch(self, epoch, trainloader, statistics_steps=1):
        running_class_loss = 0.0
        running_object_confidence_loss = 0.0
        running_no_object_confidence_loss = 0.0
        running_coord_loss = 0.0
        steps_time = 0.0
        for step, data in enumerate(trainloader, 1):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            start_time = timeit.default_timer()
            class_loss, object_confidence_loss, no_object_confidence_loss, coord_loss = self.train_step(inputs, labels)
            steps_time += timeit.default_timer() - start_time

            class_loss, object_confidence_loss, no_object_confidence_loss, coord_loss = class_loss.item(), object_confidence_loss.item(), no_object_confidence_loss.item(), coord_loss.item()

            self.logger.line("class loss", class_loss)
            self.logger.line("object confidence loss", object_confidence_loss)
            self.logger.line("no object confidence loss", no_object_confidence_loss)
            self.logger.line("coord loss", coord_loss)

            running_class_loss += class_loss
            running_object_confidence_loss += object_confidence_loss
            running_no_object_confidence_loss += no_object_confidence_loss
            running_coord_loss += coord_loss
            if step % statistics_steps == 0:
                print(f"epoch:{epoch}  "
                      f"step:{step}  "
                      f"time:{steps_time / statistics_steps:.3f}s/step  "
                      f"class loss: {running_class_loss / statistics_steps:.3f}  "
                      f"obj conf loss: {running_object_confidence_loss / statistics_steps:.3f}  "
                      f"no obj conf loss: {running_no_object_confidence_loss / statistics_steps:.3f}  "
                      f"coord loss: {running_coord_loss / statistics_steps:.3f}")
                running_class_loss = 0.0
                running_object_confidence_loss = 0.0
                running_no_object_confidence_loss = 0.0
                running_coord_loss = 0.0
                steps_time = 0.0

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
