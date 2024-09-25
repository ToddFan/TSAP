import torch
import torch.nn as nn

class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()


    def forward(self, img_batch, patch, patch_mask_tf):
    # def forward(self, img_batch, segs, patch, patch_mask_tf):
    #
    #     patch = torch.mul(segs, patch)
    #     patch_mask_tf = torch.mul(segs, patch_mask_tf)

        patch_mask = patch - 1
        patch_mask = - patch_mask  # 得到一个与补丁掩码相反的掩码，即非零值为0，零值为1。

        imgWithPatch = torch.mul(img_batch, patch_mask) + torch.mul(patch, patch_mask_tf)

        return imgWithPatch
