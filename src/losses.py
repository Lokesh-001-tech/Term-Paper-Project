import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# SSIM
# ------------------------------------------------------------
def _gaussian_window(channels, size=11, sigma=1.5):
    coords = torch.arange(size).float() - size // 2

    g = torch.exp(
        -(coords ** 2) / (2 * sigma ** 2)
    )

    g = g / g.sum()

    window = g[:, None] * g[None, :]

    return window.expand(
        channels,
        1,
        size,
        size
    ).contiguous()


def ssim(img1, img2, window_size=11):

    channels = img1.shape[1]

    window = _gaussian_window(
        channels,
        window_size
    ).to(
        img1.device,
        img1.dtype
    )

    pad = window_size // 2

    mu1 = F.conv2d(
        img1,
        window,
        padding=pad,
        groups=channels
    )

    mu2 = F.conv2d(
        img2,
        window,
        padding=pad,
        groups=channels
    )

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            padding=pad,
            groups=channels
        )
        - mu1_sq
    )

    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            padding=pad,
            groups=channels
        )
        - mu2_sq
    )

    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            padding=pad,
            groups=channels
        )
        - mu1_mu2
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = (
        (2 * mu1_mu2 + c1)
        * (2 * sigma12 + c2)
    ) / (
        (mu1_sq + mu2_sq + c1)
        * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean()


# ------------------------------------------------------------
# PSNR
# ------------------------------------------------------------
def psnr(img1, img2):

    mse = F.mse_loss(img1, img2)

    return 10 * torch.log10(
        1.0 / (mse + 1e-10)
    )


# ------------------------------------------------------------
# Sobel gradient calculation
# ------------------------------------------------------------
def sobel_gradients(image):

    channels = image.shape[1]

    kernel_x = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ],
        dtype=image.dtype,
        device=image.device
    ).view(1, 1, 3, 3)

    kernel_y = kernel_x.transpose(2, 3)

    kernel_x = kernel_x.repeat(
        channels, 1, 1, 1
    )

    kernel_y = kernel_y.repeat(
        channels, 1, 1, 1
    )

    gx = F.conv2d(
        image,
        kernel_x,
        padding=1,
        groups=channels
    )

    gy = F.conv2d(
        image,
        kernel_y,
        padding=1,
        groups=channels
    )

    return gx, gy


# ------------------------------------------------------------
# Single-scale edge loss
# ------------------------------------------------------------
def edge_loss(output, target):

    out_x, out_y = sobel_gradients(output)
    tgt_x, tgt_y = sobel_gradients(target)

    loss_x = F.l1_loss(
        out_x,
        tgt_x
    )

    loss_y = F.l1_loss(
        out_y,
        tgt_y
    )

    return loss_x + loss_y


# ------------------------------------------------------------
# Multi-scale edge loss
# ------------------------------------------------------------
def multi_scale_edge_loss(output, target):
    """
    Compares image gradients at multiple resolutions.

    Full resolution:
        Fine edges and small details.

    Half resolution:
        Medium-scale structures and textures.

    This helps prevent the network from only learning
    colour/brightness correction while leaving blurry edges.
    """

    # --------------------------------------------------------
    # Scale 1: original resolution
    # --------------------------------------------------------

    loss_full = edge_loss(
        output,
        target
    )

    # --------------------------------------------------------
    # Scale 2: half resolution
    # --------------------------------------------------------

    output_half = F.avg_pool2d(
        output,
        kernel_size=2,
        stride=2
    )

    target_half = F.avg_pool2d(
        target,
        kernel_size=2,
        stride=2
    )

    loss_half = edge_loss(
        output_half,
        target_half
    )

    # Give more importance to fine details
    return (
        0.7 * loss_full
        + 0.3 * loss_half
    )


# ------------------------------------------------------------
# Colour loss
# ------------------------------------------------------------
def color_loss(output, target):
    """
    Matches the mean and contrast of each RGB channel.
    """

    mean_loss = F.l1_loss(
        output.mean(dim=(2, 3)),
        target.mean(dim=(2, 3))
    )

    std_loss = F.l1_loss(
        output.std(dim=(2, 3)),
        target.std(dim=(2, 3))
    )

    return mean_loss + std_loss


# ------------------------------------------------------------
# VGG perceptual loss
# ------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):

    def __init__(self):

        super(VGGPerceptualLoss, self).__init__()

        from torchvision.models import (
            vgg16,
            VGG16_Weights
        )

        vgg = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        ).features[:16]

        vgg.eval()

        for parameter in vgg.parameters():
            parameter.requires_grad = False

        self.vgg = vgg

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1)
        )

    def forward(self, output, target):

        output = (
            output.clamp(0, 1)
            - self.mean
        ) / self.std

        target = (
            target
            - self.mean
        ) / self.std

        return F.l1_loss(
            self.vgg(output),
            self.vgg(target)
        )


# ------------------------------------------------------------
# Combined restoration loss
# ------------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Combined loss for underwater image restoration.

    Components:

        L1
            Pixel-level reconstruction.

        SSIM
            Structural similarity.

        Multi-scale Edge
            Fine and medium-scale edge/detail restoration.

        Colour
            RGB colour and contrast correction.

        Perceptual
            Texture and high-level visual similarity.
    """

    def __init__(
        self,
        ssim_weight=0.2,
        perceptual_weight=0.05,
        edge_weight=0.4,
        color_weight=0.2,
        use_perceptual=True
    ):

        super(CombinedLoss, self).__init__()

        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight

        self.perceptual = None

        if use_perceptual:

            try:

                self.perceptual = (
                    VGGPerceptualLoss()
                )

                print(
                    "Perceptual loss: ON"
                )

            except Exception as error:

                print(
                    "Perceptual loss: OFF "
                    "(could not load VGG16):",
                    error
                )

    def forward(self, output, target):

        # ----------------------------------------------------
        # Keep predictions in valid image range
        # ----------------------------------------------------

        output_clamped = output.clamp(
            0,
            1
        )

        # ----------------------------------------------------
        # 1. Pixel reconstruction
        # ----------------------------------------------------

        loss = F.l1_loss(
            output_clamped,
            target
        )

        # ----------------------------------------------------
        # 2. Structural similarity
        # ----------------------------------------------------

        ssim_component = (
            1
            - ssim(
                output_clamped,
                target
            )
        )

        loss = (
            loss
            + self.ssim_weight
            * ssim_component
        )

        # ----------------------------------------------------
        # 3. Multi-scale edge/detail restoration
        # ----------------------------------------------------

        edge_component = (
            multi_scale_edge_loss(
                output_clamped,
                target
            )
        )

        loss = (
            loss
            + self.edge_weight
            * edge_component
        )

        # ----------------------------------------------------
        # 4. Colour correction
        # ----------------------------------------------------

        color_component = color_loss(
            output_clamped,
            target
        )

        loss = (
            loss
            + self.color_weight
            * color_component
        )

        # ----------------------------------------------------
        # 5. Perceptual texture/detail loss
        # ----------------------------------------------------

        if self.perceptual is not None:

            perceptual_component = (
                self.perceptual(
                    output_clamped,
                    target
                )
            )

            loss = (
                loss
                + self.perceptual_weight
                * perceptual_component
            )

        return loss
