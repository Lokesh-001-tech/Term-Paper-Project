import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# SSIM (structural similarity) - used both as a loss and a metric
# ------------------------------------------------------------
def _gaussian_window(channels, size=11, sigma=1.5):
    coords = torch.arange(size).float() - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]
    return window.expand(channels, 1, size, size).contiguous()


def ssim(img1, img2, window_size=11):
    channels = img1.shape[1]
    window = _gaussian_window(channels, window_size).to(
        img1.device, img1.dtype
    )
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean()


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 10 * torch.log10(1.0 / (mse + 1e-10))


# ------------------------------------------------------------
# Edge (gradient) loss - rewards sharp edges
# ------------------------------------------------------------
def edge_loss(output, target):
    """L1 difference between Sobel gradients of output and target."""

    channels = output.shape[1]

    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=output.dtype, device=output.device
    ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)

    kernel_y = kernel_x.transpose(2, 3)

    def gradients(image):
        gx = F.conv2d(image, kernel_x, padding=1, groups=channels)
        gy = F.conv2d(image, kernel_y, padding=1, groups=channels)
        return gx, gy

    out_x, out_y = gradients(output)
    tgt_x, tgt_y = gradients(target)

    return F.l1_loss(out_x, tgt_x) + F.l1_loss(out_y, tgt_y)


# ------------------------------------------------------------
# Colour loss - matches each channel's average level and contrast
# ------------------------------------------------------------
def color_loss(output, target):
    """
    Forces the R, G, B channels to have the same overall brightness
    (mean) and contrast (std) as the reference. This fixes the
    under-corrected red channel.
    """

    mean_loss = F.l1_loss(output.mean(dim=(2, 3)), target.mean(dim=(2, 3)))
    std_loss = F.l1_loss(output.std(dim=(2, 3)), target.std(dim=(2, 3)))

    return mean_loss + std_loss


# ------------------------------------------------------------
# Perceptual loss (VGG16 features) - restores texture and sharpness
# ------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()

        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        vgg.eval()

        for parameter in vgg.parameters():
            parameter.requires_grad = False

        self.vgg = vgg

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, output, target):
        output = (output.clamp(0, 1) - self.mean) / self.std
        target = (target - self.mean) / self.std

        return F.l1_loss(self.vgg(output), self.vgg(target))


# ------------------------------------------------------------
# Combined loss
# ------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    total = L1
          + ssim_weight       * (1 - SSIM)
          + edge_weight       * edge (gradient) loss
          + color_weight      * colour (channel mean/std) loss
          + perceptual_weight * VGG perceptual loss

    L1         -> accurate colours and brightness
    SSIM       -> preserves structure
    Edge       -> sharper edges (deblurring)
    Colour     -> fully restores the red channel and contrast
    Perceptual -> natural textures (needs a GPU)
    """

    def __init__(self, ssim_weight=0.2, perceptual_weight=0.05,
                 edge_weight=0.3, color_weight=0.2,
                 use_perceptual=True):
        super(CombinedLoss, self).__init__()

        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight

        self.perceptual = None

        if use_perceptual:
            try:
                self.perceptual = VGGPerceptualLoss()
                print("Perceptual loss: ON")
            except Exception as error:
                print("Perceptual loss: OFF (could not load VGG16):", error)

    def forward(self, output, target):

        loss = F.l1_loss(output, target)

        loss = loss + self.ssim_weight * (
            1 - ssim(output.clamp(0, 1), target)
        )

        loss = loss + self.edge_weight * edge_loss(output, target)

        loss = loss + self.color_weight * color_loss(output, target)

        if self.perceptual is not None:
            loss = loss + self.perceptual_weight * self.perceptual(
                output, target
            )

        return loss