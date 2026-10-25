import inspect
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

from sr.archs.sr_arch import SuperResolutionNet

from sr.common.tile_geometry import iter_tile_regions


def _load_checkpoint(model, checkpoint_path, param_key='model_state'):
    """Strictly load trusted inference parameters without a training framework."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Model checkpoint does not exist: {checkpoint_path}')
    load_options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_options['weights_only'] = True
    checkpoint = torch.load(str(checkpoint_path), **load_options)
    if isinstance(checkpoint, dict) and param_key == 'params_ema' and param_key not in checkpoint:
        param_key = 'params'
    state = checkpoint.get(param_key) if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f'Model checkpoint has no {param_key} dictionary: {checkpoint_path}')
    state = OrderedDict(
        (name[7:] if name.startswith('module.') else name, value)
        for name, value in state.items())
    model.load_state_dict(state, strict=True)



def _image_to_tensor(image):
    """BGR HWC image to float RGB CHW; the caller controls value normalization."""
    # Preserve the original HWC -> CHW strides; changing memory layout can
    # select different CUDA convolution kernels and alter rounded SR pixels.
    rgb = image[:, :, ::-1].copy()
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def _tensor_to_image(tensor):
    """One RGB BCHW tensor in [0, 1] to BGR uint8, preserving SR rounding."""
    tensor = tensor.squeeze(0).float().detach().cpu().clamp_(0, 1)
    image = tensor.numpy().transpose(1, 2, 0)[:, :, ::-1]
    return (image * 255.0).round().astype(np.uint8)


# Checkpoint-compatible document restoration forward paths, without training code.

def _spatial_to_tokens(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def _tokens_to_spatial(x, height, width):
    return rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)


class _AffineFeatureNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class _SpatialFeatureNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = _AffineFeatureNorm(dim)

    def forward(self, x):
        height, width = x.shape[-2:]
        return _tokens_to_spatial(self.body(_spatial_to_tokens(x)), height, width)


class _GatedFeedForward(nn.Module):
    def __init__(self, dim, expansion, bias):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class _FeatureAttention(nn.Module):
    def __init__(self, dim, heads, bias):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        height, width = x.shape[-2:]
        q, k, value = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.heads)
        value = rearrange(value, 'b (head c) h w -> b head c (h w)', head=self.heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1) * self.temperature).softmax(dim=-1)
        output = attention @ value
        output = rearrange(
            output, 'b head c (h w) -> b (head c) h w',
            head=self.heads, h=height, w=width)
        return self.project_out(output)


class _RestorationBlock(nn.Module):
    def __init__(self, dim, heads, expansion, bias):
        super().__init__()
        self.norm1 = _SpatialFeatureNorm(dim)
        self.attn = _FeatureAttention(dim, heads, bias)
        self.norm2 = _SpatialFeatureNorm(dim)
        self.ffn = _GatedFeedForward(dim, expansion, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _RestorationEmbedding(nn.Module):
    def __init__(self, input_channels, embed_dim, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(input_channels, embed_dim, 3, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class _FeatureDownsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class _FeatureUpsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class _PromptRestorationNet(nn.Module):
    """Prompt-conditioned network for geometry, shadows, and foreground masks."""
    def __init__(self):
        super().__init__()
        dim = 48
        blocks = [2, 3, 3, 4]
        heads = [1, 2, 4, 8]
        self.patch_embed = _RestorationEmbedding(6, dim)
        self.encoder_level1 = nn.Sequential(*[
            _RestorationBlock(dim, heads[0], 2.66, False)
            for _ in range(blocks[0])])
        self.down1_2 = _FeatureDownsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            _RestorationBlock(dim * 2, heads[1], 2.66, False)
            for _ in range(blocks[1])])
        self.down2_3 = _FeatureDownsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[
            _RestorationBlock(dim * 4, heads[2], 2.66, False)
            for _ in range(blocks[2])])
        self.down3_4 = _FeatureDownsample(dim * 4)
        self.latent = nn.Sequential(*[
            _RestorationBlock(dim * 8, heads[3], 2.66, False)
            for _ in range(blocks[3])])
        self.up4_3 = _FeatureUpsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = nn.Sequential(*[
            _RestorationBlock(dim * 4, heads[2], 2.66, False)
            for _ in range(blocks[2])])
        self.up3_2 = _FeatureUpsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = nn.Sequential(*[
            _RestorationBlock(dim * 2, heads[1], 2.66, False)
            for _ in range(blocks[1])])
        self.up2_1 = _FeatureUpsample(dim * 2)
        self.decoder_level1 = nn.Sequential(*[
            _RestorationBlock(dim * 2, heads[0], 2.66, False)
            for _ in range(blocks[0])])
        self.refinement = nn.Sequential(*[
            _RestorationBlock(dim * 2, heads[0], 2.66, False)
            for _ in range(4)])
        self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.output = nn.Conv2d(dim * 2, 3, 3, padding=1, bias=False)

    def forward(self, image):
        embedded = self.patch_embed(image)
        level1 = self.encoder_level1(embedded)
        level2 = self.encoder_level2(self.down1_2(level1))
        level3 = self.encoder_level3(self.down2_3(level2))
        latent = self.latent(self.down3_4(level3))
        decoded3 = self.up4_3(latent)
        decoded3 = self.reduce_chan_level3(torch.cat((decoded3, level3), dim=1))
        decoded3 = self.decoder_level3(decoded3)
        decoded2 = self.up3_2(decoded3)
        decoded2 = self.reduce_chan_level2(torch.cat((decoded2, level2), dim=1))
        decoded2 = self.decoder_level2(decoded2)
        decoded1 = self.up2_1(decoded2)
        decoded1 = self.decoder_level1(torch.cat((decoded1, level1), dim=1))
        decoded1 = self.refinement(decoded1)
        return self.output(decoded1 + self.skip_conv(embedded))


class _MaskResidualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = batch_norm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=stride, padding=dilation,
            dilation=dilation, bias=False)
        self.bn2 = batch_norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = batch_norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(output + residual)


class _MaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        strides = [1, 2, 2, 1]
        dilations = [1, 1, 1, 2]
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, strides[0], dilations[0])
        self.layer2 = self._make_layer(128, 4, strides[1], dilations[1])
        self.layer3 = self._make_layer(256, 23, strides[2], dilations[2])
        self.layer4 = self._make_mg_unit(512, [1, 2, 4], strides[3], dilations[3])

    def _make_layer(self, planes, count, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4))
        layers = [_MaskResidualBottleneck(
            self.inplanes, planes, stride, dilation, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, count):
            layers.append(_MaskResidualBottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_mg_unit(self, planes, dilations, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4))
        layers = [_MaskResidualBottleneck(
            self.inplanes, planes, stride, dilations[0] * dilation, downsample)]
        self.inplanes = planes * 4
        for block_dilation in dilations[1:]:
            layers.append(_MaskResidualBottleneck(
                self.inplanes, planes, dilation=block_dilation * dilation))
        return nn.Sequential(*layers)

    def forward(self, image):
        output = self.relu(self.bn1(self.conv1(image)))
        output = self.maxpool(output)
        output = self.layer1(output)
        low_level = output
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output), low_level


class _MaskDilatedBranch(nn.Module):
    def __init__(self, dilation, batch_norm=nn.BatchNorm2d):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            2048, 256, 3 if dilation > 1 else 1,
            padding=dilation if dilation > 1 else 0,
            dilation=dilation, bias=False)
        self.bn = batch_norm(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class _MaskContextPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.aspp1 = _MaskDilatedBranch(1)
        self.aspp2 = _MaskDilatedBranch(6)
        self.aspp3 = _MaskDilatedBranch(12)
        self.aspp4 = _MaskDilatedBranch(18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = [branch(x) for branch in (self.aspp1, self.aspp2, self.aspp3, self.aspp4)]
        pooled = self.global_avg_pool(x)
        pooled = F.interpolate(
            pooled, size=outputs[-1].shape[2:], mode='bilinear', align_corners=True)
        output = torch.cat(outputs + [pooled], dim=1)
        return self.relu(self.bn1(self.conv1(output)))


class _MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            # Keep sequence indices aligned with the saved checkpoint.
            nn.BatchNorm2d(256), nn.ReLU(), nn.Identity(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Identity(),
            nn.Conv2d(256, 1, 1), nn.Sigmoid())

    def forward(self, features, low_level):
        low_level = self.relu(self.bn1(self.conv1(low_level)))
        features = F.interpolate(
            features, size=low_level.shape[2:], mode='bilinear', align_corners=True)
        return self.last_conv(torch.cat((features, low_level), dim=1))


class _DocumentMaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MaskEncoder()
        self.aspp = _MaskContextPyramid()
        self.decoder = _MaskDecoder()

    def forward(self, image):
        features, low_level = self.backbone(image)
        prediction = self.decoder(self.aspp(features), low_level)
        return F.interpolate(
            prediction, size=image.shape[2:], mode='bilinear', align_corners=True)


class _DocumentMaskPredictor:
    def __init__(self, checkpoint_path, device):
        self._cv2 = __import__('cv2')
        self._device = torch.device(device)
        self._model = _DocumentMaskNet()
        _load_checkpoint(self._model, checkpoint_path)
        self._model = self._model.float().to(self._device).eval()

    def predict(self, image):
        cv2 = self._cv2
        height, width = image.shape[:2]
        resized = cv2.resize(image, (448, 448))
        resized = cv2.GaussianBlur(resized, (15, 15), 0, 0)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).unsqueeze(0)
        tensor = tensor.to(self._device, dtype=torch.float32) / 255.0
        with torch.inference_mode():
            prediction = self._model(tensor)[:, :1]
            prediction = F.interpolate(prediction, (height, width), mode='nearest')
            mask = (prediction[0, 0].cpu().numpy() * 255).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        mask = cv2.erode(mask, kernel, iterations=3)
        mask[mask > 100] = 255
        mask[mask < 100] = 0
        return mask

    def close(self):
        self._model = None
        if self._device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


class _PerspectiveCorrector:
    INPUT_SIZE = 256

    def __init__(self, model_path, mask_model_path, device):
        self._cv2 = __import__('cv2')
        self._device = torch.device(device)
        self._model = _PromptRestorationNet()
        _load_checkpoint(self._model, model_path)
        self._model = self._model.float().to(self._device).eval()
        self._mask_predictor = _DocumentMaskPredictor(mask_model_path, self._device)

    @staticmethod
    def _validate_image(image):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError('restoration input must be a uint8 BGR image with three channels.')
        if min(image.shape[:2]) < 2:
            raise ValueError('restoration input must be at least 2 x 2 pixels.')
        if max(image.shape[:2]) >= 32767:
            raise ValueError('OpenCV remap requires image dimensions below 32767 pixels.')

    @classmethod
    def _base_coordinates(cls):
        y, x = np.mgrid[:cls.INPUT_SIZE, :cls.INPUT_SIZE]
        return np.stack(
            (x / cls.INPUT_SIZE, y / cls.INPUT_SIZE), axis=-1).astype(np.float32)

    def _build_prompt(self, image, mask):
        self._validate_image(image)
        if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]:
            raise ValueError('restoration document mask must match the input image size.')
        masked = image.copy()
        masked[mask == 0] = 0
        resized_mask = self._cv2.resize(mask, (self.INPUT_SIZE, self.INPUT_SIZE))
        resized_mask = resized_mask.astype(np.float32) / 255.0
        prompt = np.concatenate((
            self._base_coordinates(), resized_mask[:, :, None]), axis=-1)
        return masked, prompt

    def _remap(self, image, residual):
        self._validate_image(image)
        if (residual.shape != (self.INPUT_SIZE, self.INPUT_SIZE, 2)
                or not np.isfinite(residual).all()):
            raise ValueError('restoration must predict a finite 256 x 256 x 2 coordinate field.')
        height, width = image.shape[:2]
        coordinates = residual + self._base_coordinates()
        for _ in range(15):
            coordinates = self._cv2.blur(
                coordinates, (3, 3), borderType=self._cv2.BORDER_REPLICATE)
        coordinates = self._cv2.resize(coordinates, (width, height))
        coordinates = (coordinates * (width, height)).astype(np.float32)
        return self._cv2.remap(
            image, coordinates[:, :, 0], coordinates[:, :, 1], self._cv2.INTER_LINEAR)

    def dewarp(self, image):
        if self._model is None or self._mask_predictor is None:
            raise RuntimeError('PerspectiveCorrector is closed.')
        self._validate_image(image)
        mask = self._mask_predictor.predict(image)
        if not np.any(mask):
            raise ValueError('Document mask predictor did not detect a document in this image.')
        masked, prompt = self._build_prompt(image, mask)
        resized = self._cv2.resize(masked, (self.INPUT_SIZE, self.INPUT_SIZE))
        model_input = np.concatenate((
            resized.astype(np.float32) / 255.0, prompt), axis=-1)
        tensor = torch.from_numpy(
            model_input.transpose(2, 0, 1).copy()).unsqueeze(0)
        tensor = tensor.to(self._device, dtype=torch.float32)
        with torch.inference_mode():
            residual = self._model(tensor)[0, :2].permute(1, 2, 0).cpu().numpy()
        return self._remap(image, residual)

    def close(self):
        model = self._model
        self._model = None
        if model is not None:
            del model
        mask_predictor = self._mask_predictor
        self._mask_predictor = None
        if mask_predictor is not None:
            mask_predictor.close()
        if self._device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


class _ForegroundCleaner:
    INPUT_STRIDE = 8
    SHADOW_MAX_SIDE = 1600
    SHADOW_PROMPT_SIZE = 1024

    def __init__(self, model_path, device):
        import cv2
        from skimage.filters import threshold_sauvola

        self._cv2 = cv2
        self._threshold_sauvola = threshold_sauvola
        self._device = torch.device(device)
        self._dtype = torch.float16 if self._device.type == 'cuda' else torch.float32
        self._model = _PromptRestorationNet()
        _load_checkpoint(self._model, model_path)
        self._model = self._model.to(self._device, dtype=self._dtype).eval()

    @staticmethod
    def _apply_mask(image, foreground_mask):
        if (not isinstance(image, np.ndarray) or image.ndim != 3
                or image.shape[2] != 3):
            raise ValueError('SR output must be a three-channel numpy image.')
        if (not isinstance(foreground_mask, np.ndarray)
                or foreground_mask.shape != image.shape[:2]):
            raise ValueError('restoration foreground mask must match the SR output size.')
        result = np.full_like(image, 255)
        result[foreground_mask.astype(bool)] = image[foreground_mask.astype(bool)]
        return result

    def _odd_window_size(self, ratio, height, width):
        window_size = max(3, int(ratio * min(height, width)))
        return window_size if window_size % 2 else window_size + 1

    def _build_prompt(self, image):
        cv2 = self._cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        first_threshold = self._threshold_sauvola(
            gray, window_size=self._odd_window_size(0.05, height, width), k=0.5)
        denominator = float(np.max(gray)) - first_threshold
        contrast = np.zeros_like(first_threshold, dtype=np.float32)
        bright = gray > first_threshold
        np.divide(gray - first_threshold, denominator, out=contrast,
                  where=bright & (denominator != 0))
        contrast = np.clip(contrast * 255.0, 0, 255).astype(np.uint8)
        second_threshold = self._threshold_sauvola(
            contrast, window_size=self._odd_window_size(0.10, height, width), k=0.5)
        binary = np.where(contrast > second_threshold, 255, 0).astype(np.uint8)
        threshold = np.clip(second_threshold, 0, 255).astype(np.uint8)
        gradient_x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        gradient_y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        gradient = cv2.addWeighted(
            cv2.convertScaleAbs(gradient_x), 0.5,
            cv2.convertScaleAbs(gradient_y), 0.5, 0)
        gradient = cv2.cvtColor(gradient, cv2.COLOR_BGR2GRAY)
        return np.stack((threshold, gradient, binary), axis=-1)

    def _pad_to_stride(self, image):
        height, width = image.shape[:2]
        padding_height = (-height) % self.INPUT_STRIDE
        padding_width = (-width) % self.INPUT_STRIDE
        if padding_height or padding_width:
            image = self._cv2.copyMakeBorder(
                image, padding_height, 0, padding_width, 0,
                borderType=self._cv2.BORDER_REPLICATE)
        return image, padding_height, padding_width

    def predict_foreground_mask(self, image):
        """Locate candidate text/foreground without changing the source pixels."""
        if self._model is None:
            raise RuntimeError('ForegroundCleaner is closed.')
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError('SR output must be a uint8 three-channel image.')
        padded, padding_height, padding_width = self._pad_to_stride(image)
        prompt = self._build_prompt(padded)
        model_input = np.concatenate((padded, prompt), axis=-1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(
            model_input.transpose(2, 0, 1)).unsqueeze(0).to(
                self._device, dtype=self._dtype, non_blocking=True)
        with torch.inference_mode():
            logits = self._model(tensor)[:, :2]
            foreground = torch.argmax(logits, dim=1)[0].eq(0).cpu().numpy()
        if foreground.shape != padded.shape[:2]:
            foreground = self._cv2.resize(
                foreground.astype(np.uint8), padded.shape[:2][::-1],
                interpolation=self._cv2.INTER_NEAREST).astype(bool)
        foreground = foreground[padding_height:, padding_width:]
        return foreground

    def remove_background(self, image):
        return self._apply_mask(image, self.predict_foreground_mask(image))

    def _build_shadow_prompt(self, image):
        """DocRes deshadowing prompt; this is conditioning, not the final output."""
        cv2 = self._cv2
        resized = cv2.resize(image, (self.SHADOW_PROMPT_SIZE, self.SHADOW_PROMPT_SIZE))
        kernel = np.ones((7, 7), dtype=np.uint8)
        background = np.stack([
            cv2.medianBlur(cv2.dilate(channel, kernel), 21)
            for channel in cv2.split(resized)
        ], axis=-1)
        return cv2.resize(background, image.shape[1::-1])

    def remove_shadow(self, image):
        """Run the checkpoint's native deshadowing task on the complete page.

        Follows DocRes inference.py: BGR image + background prompt, stride-8
        padding, and a model-predicted illumination map for large originals.
        """
        if self._model is None:
            raise RuntimeError('ForegroundCleaner is closed.')
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
            raise ValueError('Shadow removal requires a uint8 BGR image of at least 2 x 2 pixels.')
        cv2 = self._cv2
        height, width = image.shape[:2]
        prompt = self._build_shadow_prompt(image)
        model_input = np.concatenate((image, prompt), axis=-1)
        resized = max(height, width) >= self.SHADOW_MAX_SIDE
        if resized:
            model_input = cv2.resize(model_input, (self.SHADOW_MAX_SIDE, self.SHADOW_MAX_SIDE))
            pad_height = pad_width = 0
        else:
            model_input, pad_height, pad_width = self._pad_to_stride(model_input)
        model_input = model_input.astype(np.float32) / 255.0
        tensor = torch.from_numpy(model_input.transpose(2, 0, 1)).unsqueeze(0).to(
            self._device, dtype=self._dtype, non_blocking=True)
        with torch.inference_mode():
            prediction = self._model(tensor)
            if tuple(prediction.shape) != (1, 3, *model_input.shape[:2]):
                raise ValueError('Shadow removal model must return a three-channel image at input size.')
            restored = prediction[0].float().cpu().numpy().transpose(1, 2, 0)
        restored = (np.clip(restored, 0, 1) * 255.0).astype(np.uint8)
        restored = restored[pad_height:, pad_width:]
        if resized:
            # The illumination comes from the model output, not from the prompt.
            working = cv2.resize(image, (self.SHADOW_MAX_SIDE, self.SHADOW_MAX_SIDE))
            illumination = working.astype(np.float64) / np.maximum(restored, 1)
            illumination = cv2.resize(illumination, (width, height))
            restored = image.astype(np.float64) / np.clip(illumination, 1e-5, None)
        return np.clip(restored, 0, 255).astype(np.uint8)

    def enhance_text(self, image, strength=0.5):
        """Darken candidate text in its original colors; copy all other pixels."""
        if (isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not np.isfinite(strength) or not 0 <= strength <= 1):
            raise ValueError('Text enhancement strength must be between 0 and 1.')
        mask = self.predict_foreground_mask(image)
        result = image.copy()
        # One multiplier for BGR preserves channel ratios, unlike black fill.
        result[mask] = np.rint(image[mask].astype(np.float32) * (1 - 0.65 * strength)).astype(np.uint8)
        return result

    def fade_background(self, image, strength=0.5):
        """Lighten original background pixels, preserving predicted foreground."""
        if (isinstance(strength, bool) or not isinstance(strength, (int, float))
                or not np.isfinite(strength) or not 0 <= strength <= 1):
            raise ValueError('Background fading strength must be between 0 and 1.')
        result = image.copy()
        if strength == 0:
            return result
        background = ~self.predict_foreground_mask(image)
        pixels = image[background].astype(np.float32)
        # Retain some original color even at full strength; never replace with white.
        result[background] = np.rint(pixels + (255 - pixels) * (0.85 * strength)).astype(np.uint8)
        return result

    def close(self):
        model = self._model
        self._model = None
        if model is not None:
            del model
        if self._device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


class RestorationPipeline:
    """Standalone inference pipeline; owns SR and optional restoration networks."""

    def __init__(self, options):
        self.opt = options
        self.device = torch.device('cuda' if options['num_gpu'] else 'cpu')
        self.net_g = None
        if options.get('use_super_resolution', True):
            self.net_g = SuperResolutionNet()
            _load_checkpoint(self.net_g, options['weight'], param_key='params_ema')
            self.net_g = self.net_g.to(self.device).eval().requires_grad_(False)
            if options['num_gpu'] > 1:
                self.net_g = nn.DataParallel(self.net_g, device_ids=list(range(options['num_gpu'])))


    def clear_attention_mask_cache(self):
        network = self.net_g.module if isinstance(self.net_g, nn.DataParallel) else self.net_g
        if network is not None:
            network.clear_attention_mask_cache()

    def _get_background_remover(self):
        options = next((self.opt[name] for name in (
            'shadow_removal', 'background_removal', 'text_enhancement', 'background_fading')
            if self.opt.get(name, {}).get('enabled', False)), None)
        if options is None:
            return None
        remover = getattr(self, '_background_remover', None)
        if remover is None:
            remover = _ForegroundCleaner(
                model_path=options['weight'],
                device=self.device,
            )
            self._background_remover = remover
        return remover

    def close_background_remover(self):
        remover = getattr(self, '_background_remover', None)
        self._background_remover = None
        if remover is not None:
            remover.close()

    def _get_perspective_corrector(self):
        options = self.opt.get('dewarping')
        if not options or not options.get('enabled', False):
            return None
        dewarper = getattr(self, '_perspective_corrector', None)
        if dewarper is None:
            dewarper = _PerspectiveCorrector(
                model_path=options['weight'],
                mask_model_path=options['mask_weight'],
                device=self.device,
            )
            self._perspective_corrector = dewarper
        return dewarper

    def close_perspective_corrector(self):
        dewarper = getattr(self, '_perspective_corrector', None)
        self._perspective_corrector = None
        if dewarper is not None:
            dewarper.close()

    def _apply_dewarping(self, image):
        dewarper = self._get_perspective_corrector()
        if dewarper is None:
            return image
        corrected_image = dewarper.dewarp(image)
        if corrected_image.shape != image.shape:
            raise ValueError(
                'restoration dewarping must preserve the input image height, width, and channels.')
        return corrected_image

    def _apply_foreground(self, image, progress_callback=None):
        cleaner = self._get_background_remover()
        if cleaner is None:
            return image
        height, width = image.shape[:2]
        # Complete each full-image stage before segmenting for the next one.
        for stage, method_name in (
                ('background_fading', 'fade_background'),
                ('text_enhancement', 'enhance_text'),
                ('background_removal', 'remove_background')):
            settings = self.opt.get(stage, {})
            if not settings.get('enabled', False):
                continue
            transform = getattr(cleaner, method_name)
            arguments = {} if stage == 'background_removal' else {
                'strength': settings.get('strength', 0.5)}
            output = np.empty_like(image)
            for region in iter_tile_regions(
                    height, width, self.opt['tile']['tile_size'], self.opt['tile']['tile_pad'], 1):
                tile = image[
                    region.input_start_y_pad:region.input_end_y_pad,
                    region.input_start_x_pad:region.input_end_x_pad]
                restored = transform(tile, **arguments)
                output[
                    region.output_start_y:region.output_end_y,
                    region.output_start_x:region.output_end_x] = restored[
                        region.tile_output_start_y:region.tile_output_end_y,
                        region.tile_output_start_x:region.tile_output_end_x]
                if progress_callback is not None:
                    progress_callback()
            image = output
        return image

    def _apply_shadow_removal(self, image):
        if not self.opt.get('shadow_removal', {}).get('enabled', False):
            return image
        output = self._get_background_remover().remove_shadow(image)
        if output.shape != image.shape or output.dtype != np.uint8:
            raise ValueError('Shadow removal must preserve the uint8 image size and channels.')
        return output

    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        # Reflection requires each padding amount to be smaller than the image.
        # Keep the existing mode for normal images; support tiny API inputs too.
        padding_mode = 'reflect' if self.mod_pad_h < h and self.mod_pad_w < w else 'replicate'
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), padding_mode)

    def tile_process(self, progress_callback=None):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        _, _, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        # Assemble on CPU so subsequent stages do not retain a full GPU output.
        self.output = np.empty((output_height, output_width, 3), dtype=np.uint8)
        # The caller owns progress rendering.
        network = self.net_g
        network.eval()
        regions = iter_tile_regions(
            height, width, self.opt['tile']['tile_size'],
            self.opt['tile']['tile_pad'], self.scale)
        with torch.inference_mode():
            for region in regions:
                input_tile = self.img[
                    :, :,
                    region.input_start_y_pad:region.input_end_y_pad,
                    region.input_start_x_pad:region.input_end_x_pad,
                ]
                output_tile = _tensor_to_image(network(input_tile))
                self.output[
                    region.output_start_y:region.output_end_y,
                    region.output_start_x:region.output_end_x,
                ] = output_tile[
                    region.tile_output_start_y:region.tile_output_end_y,
                    region.tile_output_start_x:region.tile_output_end_x,
                ]
                if progress_callback is not None:
                    progress_callback()

    def post_process(self):
        h, w = self.output.shape[:2]
        self.output = self.output[:h - self.mod_pad_h * self.scale, :w - self.mod_pad_w * self.scale]

    def infer_array(self, image, progress_callback=None):
        """Run the complete document pipeline on one BGR uint8 NumPy image."""
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError('Input must be a uint8 BGR image with three channels.')
        if min(image.shape[:2]) < 2:
            raise ValueError('Input image must be at least 2 x 2 pixels.')
        try:
            use_sr = self.opt.get('use_super_resolution', True)
            output_image = image.copy()
            if use_sr:
                self.lq = _image_to_tensor(image.astype(np.float32) / 255.0).unsqueeze(0).to(self.device)
                self.pre_process()
                self.tile_process(progress_callback)
                self.post_process()
                output_image = self.output
                del self.lq, self.img, self.output
            output_image = self._apply_dewarping(output_image)
            output_image = self._apply_shadow_removal(output_image)
            return self._apply_foreground(output_image, None if use_sr else progress_callback)
        finally:
            for name in ('lq', 'img', 'output'):
                if hasattr(self, name):
                    delattr(self, name)
