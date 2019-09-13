

class Train(object):
    def __init__(self, model, optimizer, loss_func, epochs):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.epochs = epochs

    def fit(self, dataloader):
        print("train")
        for epoch in range(self.epochs):
            epoch += 1
            for step, (images, labels) in dataloader:
                step += 1
                predicts = self.model(images)
                loss = self.loss_func(labels, predicts)

