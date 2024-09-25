import argparse
import ast
import os

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from ptop import ParticleToPatch
from utils_patch import PatchApplier
from victim_detector.models.yolo import DetectionModel
from victim_detector.utils.augmentations import letterbox
from victim_detector.utils.datasets import create_dataloader
from victim_detector.utils.general import check_img_size, check_dataset

if __name__ == '__main__':
    # 读取txt内容
    with open(r'runs/result/best_solution.txt', 'r', encoding='UTF-8') as f:
        txt_content = f.read()
    start_index = txt_content.index("best_solution:")
    end_index = txt_content.index("best_score:")
    best_solution_str = txt_content[start_index + len("best_solution:"):end_index].strip()
    best_solution_list = ast.literal_eval(best_solution_str)
    best_solution = torch.Tensor(best_solution_list).cuda()

    import multiprocessing

    multiprocessing.freeze_support()

    batch_size = 128
    device = 'cuda:0'

    ptp = ParticleToPatch(10)
    pa = PatchApplier()

    # Load model
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='victim_detector/data/custom.yaml', help='dataset.yaml path')
    opt = parser.parse_args()
    data_dict = check_dataset(opt.data)
    print(data_dict)
    train_path, val_path = data_dict['train'], data_dict['val']
    hyp = 'victim_detector/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    weights = "runs/car_h_0.1_wh_1.5/weights/best.pt"
    ckpt = torch.load(weights, map_location='cpu')
    model = DetectionModel(ckpt['model'].yaml, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(640, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    dataloader = create_dataloader(val_path,
                                   imgsz,
                                   batch_size,
                                   gs)[0]

    for batch_i, (im, targets, paths, shapes) in enumerate(dataloader):
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        # im = im.half() if half else im.float()  # uint8 to fp16/32
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        seg_root = "victim_detector/path/car_h_0.1_wh_1.5/seg/val"
        seg_list = []
        for path in paths:
            filename = path.split('\\')[-1]
            seg_path = os.path.join(seg_root, filename)
            seg = Image.open(seg_path)

            # 转换为NumPy数组
            seg_array = np.array(seg, dtype=np.float32) / 255.0

            seg_array[seg_array >= 0.5] = 1
            seg_array[seg_array < 0.5] = 0

            seg_array, _, _ = letterbox(seg_array, 640, color=(0, 0, 0), auto=False, scaleup=False)

            if seg_array.shape == (640, 640, 3):
                seg_array = np.transpose(seg_array, (2, 0, 1))
            else:
                seg_array = np.repeat(seg_array[np.newaxis, :, :], 3, axis=0)
            # 转换为张量
            seg = torch.from_numpy(seg_array)
            seg_list.append(seg)
        segs = torch.stack(seg_list).to(device)

        patch_tf, patch_mask_tf = ptp(best_solution, targets, im)

        imgWithPatch = pa(im, segs,patch_tf, patch_mask_tf)

        save_dir = 'runs/result/output'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 遍历imgWithPatch中的每个照片
        for i, single_img in enumerate(tqdm(imgWithPatch)):
            # 获取im的文件名（假设im是从paths中获取的）
            base_name = os.path.basename(paths[i])  # 替换为实际的获取文件名的方法
            # 获取文件名（不包含扩展名）
            img_name = os.path.splitext(base_name)[0]
            # 构建保存路径（假设保存在output文件夹下）
            save_path = os.path.join(save_dir, f'{img_name}.jpg')  # 替换为实际的保存路径

            # print(img_name)

            # 检查文件是否存在
            if os.path.exists(save_path):
                # 删除已存在的文件
                os.remove(save_path)

            # 将single_img转换为PIL图像对象
            from torchvision import transforms

            single_img_pil = transforms.ToPILImage()(single_img.cpu())

            if len(img_name)>13:
                single_img_pil = single_img_pil.crop((0, 64, 640, 576))  # 原图尺寸（640，512） 
            else:
                single_img_pil = single_img_pil.crop((0, 80, 640, 560))  # 原图尺寸（640，480）

            # 保存为图片文件
            single_img_pil.save(save_path)
    print("Save imgWithPatch to", save_dir)
