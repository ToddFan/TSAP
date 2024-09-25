import torch

from victim_detector.models.yolo import DetectionModel
import argparse
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from victim_detector.models.common import *
from victim_detector.utils.general import (LOGGER, check_dataset, check_file, check_git_status, check_img_size,
                                           check_requirements,
                                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                                           init_seeds,
                                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                                           one_cycle,
                                           print_args, print_mutation, strip_optimizer)
from victim_detector.utils.datasets import create_dataloader
from victim_detector.utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, \
    torch_distributed_zero_first

from utils_patch import *
from DE import OptimizeFunction, DE

WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='victim_detector/models/yolov5s.yaml', help='model.yaml')

    parser.add_argument('--batch_size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')

    parser.add_argument('--data', type=str, default=ROOT / 'victim_detector/data/custom.yaml', help='dataset.yaml path')

    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')

    parser.add_argument('--epochs', type=int, default=3)

    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')

    parser.add_argument('--patch_size', type=float, default=0.2)

    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    hyp = 'victim_detector/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    weights = "runs/car_h_0.1_wh_1.5/weights/best.pt"
    ckpt = torch.load(weights, map_location='cpu')
    model = DetectionModel(ckpt['model'].yaml, ch=3, nc=1, anchors=hyp.get('anchors')).to(device)
    exclude = ['anchor'] if (hyp.get('anchors')) and not opt.resume else []
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
    model.load_state_dict(csd, strict=False)

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    # Batch size
    batch_size = opt.batch_size

    # Dataloader
    data_dict = check_dataset(opt.data)
    print(data_dict)
    train_path, val_path = data_dict['train'], data_dict['val']
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              opt.single_cls,
                                              hyp=hyp,
                                              augment=False,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=opt.workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)

    model.eval()

    nb = len(train_loader)

    # DE
    patch_num = 7
    de = DE(100, device)  # DE对象
    func = OptimizeFunction(model, opt.patch_size, device, patch_num)  # OptimizeFunction对象

    save_dir = Path('results')
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    best_fitness = 100.0

    for epoch in tqdm(range(opt.epochs),
                      desc='epoch----->'):  # epoch ------------------------------------------------------------------
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc='epoch_batch------->')

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            targets = targets.to(device)
            imgs = imgs.to(device, non_blocking=True).float() / 255  # (n, c, h, w)
            seg_root = "victim_detector/path/car_h_0.1_wh_1.5/seg/train"  # 分割掩码储存位置
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

                if seg_array.shape==(640,640,3):
                    seg_array = np.transpose(seg_array, (2, 0, 1))
                else:
                    seg_array = np.repeat(seg_array[np.newaxis, :, :], 3, axis=0)
                # 转换为张量
                seg = torch.from_numpy(seg_array)
                seg_list.append(seg)
            segs = torch.stack(seg_list).to(device)



            func.set_para(targets, imgs, segs)
            de.optimize(func)
            de.run()


