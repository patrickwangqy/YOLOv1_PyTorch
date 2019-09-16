import config
from utils import image_util


class Validation(object):
    def __init__(self, model):
        self.model = model

    def val(self, testloader):
        for inputs, _labels in testloader:
            image = image_util.tensor_to_image(inputs[0])
            inputs = inputs.to(config.device)
            predicts = self.model(inputs)
            image_util.draw_predict(image, predicts[0])
            image_util.imshow(image)
