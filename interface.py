import torch
import numpy as np
import os
import sys
import time
import cv2
from mmcv import Config
from PIL import Image
import torchvision.transforms as transforms
from models import build_model
from models.utils import fuse_module

def scale_aligned_short(img, short_size=736):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def prepare_data(img):

    # img = img[:, :, [2, 1, 0]]
    # print(img.shape)
    img_meta = dict(
        org_img_size=np.array(img.shape[:2])
    )

    img = scale_aligned_short(img)
    img_meta.update(dict(
        img_size=np.array(img.shape[:2])
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
    img = img.view(1,img.shape[0],img.shape[1],img.shape[2])
    # print(img.shape)
    data = dict(
        imgs=img,
        img_metas=img_meta
    )

    return data

def setConfig():
    config = './PSENet/config/psenet/psenet_r50_ctw_finetune.py'
    checkpoint = './PSENet/checkpoints/psenet_ctw_finetune/checkpoint.pth.tar'
    cfg = Config.fromfile(config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed = False
        ))
    sys.stdout.flush()


    # model
    model = build_model(cfg.model)
    model = model.cuda()

    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)
    model.eval()
    return model, cfg

def Visualize(model , cfg , img):

    # model.eval()
    data = prepare_data(img)

    # prepare input
    data['imgs'] = data['imgs'].cuda()
    data.update(dict(
        cfg=cfg
    ))
    # forward
    with torch.no_grad():
        outputs = model(**data)
        bboxes = outputs['bboxes']

    lines = []
    cnt = 0
    # img = img[:,:,[2,1,0]]
    for i, bbox in enumerate(bboxes):
        bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1,2)[:,[1,0]]
        cv2.polylines(img,[bbox],True,(255,0,0),lineType=cv2.LINE_AA)
        cv2.putText(img,str(cnt),(bbox[0,0],bbox[0,1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cnt += 1

    return img

def display(model, cfg):
    model.eval()
    # print(data)
    sys.stdout.flush()
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        # img = cv2.imread(r"D:\graduated design\src\PSENet\data\ctw1500\test\text_image\1001.jpg")
        flag, img = capture.read()
        img = cv2.resize(img,(int(img.shape[1] * 1.5), int(img.shape[0] * 1.5)))
        img=cv2.flip(img,1)
        shape = img.shape
        data = prepare_data(img)

        # prepare input
        data['imgs'] = data['imgs'].cuda()
        data.update(dict(
            cfg=cfg
        ))
        st = time.time()
        # forward
        with torch.no_grad():
            outputs = model(**data)


        bboxes = outputs['bboxes']

        lines = []
        cnt = 0
        for i, bbox in enumerate(bboxes):
            bbox = bbox.reshape(-1, 2)[:, ::-1].reshape(-1,2)[:,[1,0]]
            cv2.polylines(img,[bbox],True,(255,0,0),lineType=cv2.LINE_AA)
            cv2.putText(img,str(cnt),(bbox[0,0],bbox[0,1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cnt += 1
        en = time.time()
        fps = 1 / (en - st)
        cv2.putText(img, "FPS : %.2f" % fps , (shape[1] - 150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1)
        cv2.imshow('sample',img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break


def main(config,checkpoint):
    cfg = Config.fromfile(config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed = False
        ))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)


    # model
    model = build_model(cfg.model)
    model = model.cuda()

    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)

    # real-time video
    ## display(model, cfg)


    # interface
    img = cv2.imread(r"../../images/samples/local_4.0_8_8/1602.jpg")   # input image
    img = Visualize(model, cfg , img)  # transfer src to dst
    cv2.imshow("sample",img)
    c = cv2.waitKey(0)
    if c == ord('s'):
        cv2.imwrite('outputs/1602_sample.png',img)    ### save image


if __name__ == '__main__':

    ### set config
    config = './config/psenet/psenet_r50_ctw_finetune.py'
    checkpoint = './checkpoints/psenet_ctw_finetune/checkpoint.pth.tar'


    main(config,checkpoint)
