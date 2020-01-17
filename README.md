# YOLO Pytorch implements

## Train

```shell
python main.py train --data_root VOCdevkit/VOC2007 --epochs 200
```

## Detect

```shell
python main.py val --data_root VOCdevkit\\VOC2007 --checkpoint checkpoints/0200.pt
```
