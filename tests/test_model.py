from torch.utils.data.dataloader import DataLoader

from data.datasets.voc import VOCDataset
from data.transforms.yolo import YOLOTransform
from models.detect_model import DetectModel
from utils import image_util


def main():
    model = DetectModel()
    voc_ds = VOCDataset("/mnt/data/datasets", "2007")
    yolo_trans = YOLOTransform(voc_ds.classes_list())
    val_ds = voc_ds.build_dataset("val", image_transform=yolo_trans.image_transform, target_transform=yolo_trans.target_transform)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)
    for images, labels in val_dl:
        predicts = model(images)
        image = image_util.tensor_to_image(images[0])
        label = labels[0].numpy()
        predict = predicts[0].detach().numpy()
        to_show = image.copy()
        image_util.draw_label(to_show, label)
        image_util.imshow(to_show)
        to_show = image.copy()
        image_util.draw_predict(to_show, predict, threshold=0)
        image_util.imshow(to_show)
        break


if __name__ == "__main__":
    main()
