import torch
import torch.nn.functional as F
from scipy.ndimage import rotate as scipy_rotate

class ParticleToPatch:
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, x, targets, imgs):
        bs, c, h, w = imgs.size()

        patch_tmp = torch.zeros_like(imgs).cuda()
        patch_mask_tmp = torch.zeros_like(imgs).cuda()

        for i in range(targets.size(0)):
            img_idx = targets[i][0]
            bbox_w = targets[i][-2] * w
            bbox_h = targets[i][-1] * h
            bbox_x = targets[i][-4] * w
            bbox_y = targets[i][-3] * h
            bbox_tl_x = bbox_x - bbox_w / 2
            bbox_tl_y = bbox_y - bbox_h / 2

            patch_vertex_x = []
            patch_vertex_y = []

            x[:, 2][x[:, 2] < 0.5] = 0.1
            x[:, 2][x[:, 2] >= 0.5] = 0.9
            patch_pixval = x[:, 2]
            len_r = x[0, 3] / 100
            patch_l = []
            patch_w = []
            patch_a = x[:, -1]

            for j in range(self.dimension):
                vertex_x = bbox_tl_x + x[j][0] * bbox_w
                vertex_y = bbox_tl_y + x[j][1] * bbox_h
                vertex_x = max(bbox_tl_x, min(vertex_x, bbox_tl_x + bbox_w))
                vertex_y = max(bbox_tl_y, min(vertex_y, bbox_tl_y + bbox_h))

                patch_vertex_x.append(vertex_x)
                patch_vertex_y.append(vertex_y)

                p_l = bbox_h * len_r if bbox_w > bbox_h else bbox_w * len_r
                p_w = p_l * 0.45 if patch_pixval[j] == 0.1 else p_l * 0.74
                p_l = max(1, p_l)
                p_w = max(1, p_w)

                patch_l.append(p_l)
                patch_w.append(p_w)

                patch_mask = torch.ones((3, int(patch_l[j]), int(patch_w[j])), dtype=torch.float32).cuda() * patch_pixval[j]

                # 将补丁旋转到所需角度
                patch_mask_cpu = patch_mask.cpu()
                patch_mask_np = patch_mask_cpu.numpy()
                patch_mask_np_rotated = scipy_rotate(patch_mask_np, 360-int(patch_a[j].cpu()), axes=(1, 2), reshape=True, order=1)
                patch_mask_rotated = torch.from_numpy(patch_mask_np_rotated).cuda()

                patch_size_h = patch_mask_rotated.size()[-2]
                patch_size_w = patch_mask_rotated.size()[-1]
                padding_h = h - patch_size_h
                padding_w = w - patch_size_w

                if patch_vertex_x[j] + patch_l[j] > bbox_tl_x + bbox_w:
                    patch_vertex_x[j] = bbox_tl_x + bbox_w - patch_l[j]
                if patch_vertex_y[j] + patch_w[j] > bbox_tl_y + bbox_h:
                    patch_vertex_y[j] = bbox_tl_y + bbox_h - patch_w[j]

                x_center_i = int(patch_vertex_x[j] + patch_size_w / 2)
                y_center_i = int(patch_vertex_y[j] + patch_size_h / 2)
                if x_center_i - patch_size_w / 2 < bbox_tl_x:
                    x_center_i = int(bbox_tl_x + patch_size_w / 2)
                if x_center_i + patch_size_w / 2 > bbox_tl_x + bbox_w:
                    x_center_i = int(bbox_tl_x + bbox_w - patch_size_w / 2) - 1

                padding_left = x_center_i - int(0.5 * patch_size_w)
                padding_right = padding_w - padding_left
                padding_top = y_center_i - int(0.5 * patch_size_h)
                padding_bottom = padding_h - padding_top

                patch_mask_padding = F.pad(patch_mask_rotated, (padding_left, padding_right, padding_top, padding_bottom))
                patch_padding = torch.zeros_like(patch_mask_padding).cuda()
                patch_padding[patch_mask_padding != 0] = 1

                patch_tmp[int(img_idx.item())] += patch_padding
                patch_mask_tmp[int(img_idx.item())] += patch_mask_padding

        patch_tf = patch_tmp
        patch_mask_tf = patch_mask_tmp

        patch_tf.data.clamp_(0, 1)

        return patch_tf, patch_mask_tf
