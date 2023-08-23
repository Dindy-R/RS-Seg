from models.segformer import SegFormer
from models.hrnet import HRNet
from models.self_deeplab.deeplabv3P import Deeplab
import segmentation_models_pytorch as smp

custom_models = {
    'segformer': SegFormer,
    'hrnet': HRNet,
    'deeplab': Deeplab
}


def create_model(cfg: dict):
    model_type = cfg['type']
    arch = cfg['arch']
    if model_type == 'smp':
        # smp_net = getattr(smp, arch)

        encoder = cfg.get('encoder', 'resnet34')
        pretrained = cfg.get('pretrained', 'imagenet')
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        aux_params = cfg.get('aux_params', None)
        # encoder_output_stride = cfg.get('encoder_output_stride', 16)


        # model = smp_net(               # smp.UnetPlusPlus
        #     encoder_name=encoder,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights=pretrained,     # use `imagenet` pretrained weights for encoder initialization
        #     in_channels=in_channel,     # model input channels (1 for grayscale images, 3 for RGB, etc.)
        #     classes=out_channel,     # model output channels (number of classes in your datasets)
        #     aux_params=aux_params
        # )
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=in_channel,
            classes=out_channel,
            aux_params=aux_params,
            # encoder_output_stride=encoder_output_stride,
        )

    elif model_type == 'custom':
        if arch == 'segformer' or arch == 'hrnet':
            assert arch.lower() in custom_models.keys()
            net = custom_models[arch.lower()]

            encoder = cfg.get('encoder', 'mit_b0')
            pretrained = cfg.get('pretrained', None)
            in_channel = cfg.get('in_channel', 3)
            out_channel = cfg.get('out_channel', 2)
            activation = cfg.get('activation', None)

            model = net(
                encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=pretrained,  # use `imagenet` pretrained weights for encoder initialization
                in_channels=in_channel,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=out_channel,  # model output channels (number of classes in your datasets)
                activation=activation
                )

        elif arch == 'deeplab':
            assert arch.lower() in custom_models.keys()
            net = custom_models[arch.lower()]

            backbone = cfg.get('backbone', 'resnest')
            pretrained = cfg.get('pretrained', None)
            out_channel = cfg.get('num_classes', 3)
            activation = cfg.get('activation', None)

            model = net(
                backbone=backbone,
                encoder_weights=pretrained,
                num_classes=out_channel,
                activation=activation
            )

    return model


# if __name__ == '__main__':
#     from torchsummary import summary
#
#     # backbone = {'type': 'mit_b1'}
#     # seg_head = {
#     #     'type': 'SegFormerHead',
#     #     'in_channels': [64, 128, 320, 512],  # b1
#     #     'feature_strides': [4, 8, 16, 32],
#     #     'channels': 128,
#     #     'dropout_ratio': 0.1,
#     #     'num_classes': 2,
#     #     'decoder_params': {'embed_dim': 256}  # b1
#     # }
#     # pretrained = '../model_data/mit_b1.pth'
#     # model = SegFormer(backbone=backbone, decode_head=seg_head, pretrained=pretrained).cuda()
#     # summary(model, input_size=(3, 512, 512))
#
#     # for name, param in model.named_parameters(recurse=False):
#     #     print(name)
#     #
#     # for child_name, child_mod in model.named_children():
#     #     print('child:', child_name)
#     #     print(child_mod)
#
#     from segformer import SegFormer
#     from models.self_deeplab.deeplabv3P import Deeplab
#
#     encoder = 'mit_b1'
#
#     pretrained = 'imagenet'
#     in_channel = 3
#     out_channel = 2
#     activation = None
#     model = SegFormer(
#         encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights=pretrained,  # use `imagenet` pretrained weights for encoder initialization
#         in_channels=in_channel,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
#         classes=out_channel,  # model output channels (number of classes in your datasets)
#         activation=activation,
#     ).cuda()
#
#
#     summary(model, input_size=(3, 512, 512))
