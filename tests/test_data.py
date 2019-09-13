from torch.utils.data.dataloader import DataLoader

from data.datasets.voc import VOCDataset
from data.transforms.identity import IdentityTransform
import utils.image_util as image_util


def main():
    voc_ds = VOCDataset("/mnt/data/datasets", "2007")
    id_trans = IdentityTransform(voc_ds.classes_list())
    val_ds = voc_ds.build_dataset("val", image_transform=id_trans.image_transform, target_transform=id_trans.target_transform)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)
    for images, labels in val_dl:
        image = images[0].numpy()
        print(image.shape)
        bboxes = labels[0].numpy()
        image_util.draw_bbox(image, bboxes)
        image_util.imshow(image)
        break


if __name__ == "__main__":
    main()
