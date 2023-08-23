from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base.modules import Activation

from models.segformer.head import SegFormerHead

CONFIG = dict(
    strides=[4, 8, 16, 32],
    dropout_ratio=0.1,
    mit_b0={
        'decoder_channels': [32, 64, 160, 256],
        'decoder_embed_dim': 256
    },
    mit_b1={
        'decoder_channels': [64, 128, 320, 512],
        'decoder_embed_dim': 256
    },
    mit_b2={
        'decoder_channels': [64, 128, 320, 512],
        'decoder_embed_dim': 256
    },
    mit_b3={
        'decoder_channels': [64, 128, 320, 512],
        'decoder_embed_dim': 768
    },
    mit_b4={
        'decoder_channels': [64, 128, 320, 512],
        'decoder_embed_dim': 768
    },
    mit_b5={
        'decoder_channels': [64, 128, 320, 512],
        'decoder_embed_dim': 768
    },
)


class SegFormer(nn.Module):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(
        self,
        encoder_name: str = "mit_b0",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
    ):
        super().__init__()
        assert encoder_name in CONFIG.keys()
        # backbone
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        # head
        decoder_params = CONFIG[encoder_name]
        self.decode_head = SegFormerHead(
            feature_strides=CONFIG['strides'],
            in_channels=self.encoder.out_channels[2:],  # eg. mit_b0.out_channels (3, 0, 32, 64, 160, 256)
            num_classes=classes,
            embedding_dim=decoder_params['decoder_embed_dim'],
            dropout_ratio=CONFIG['dropout_ratio']
        )
        self.activation = Activation(activation)

    def forward(self, inputs):
        h, w = inputs.size(2), inputs.size(3)

        x = self.encoder(inputs)
        x = self.decode_head(x[2:])

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    model = SegFormer(encoder_name="mit_b0", encoder_depth=5, encoder_weights=None, in_channels=3, classes=6)
    model.eval()
    input = torch.rand(4, 3, 256, 256)
    output = model(input)
    print(output.size())
