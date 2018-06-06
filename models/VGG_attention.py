import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = ['vgg_attention',]


class VGG_Attention(nn.Module):

    def __init__(self, Embed_dim=64, num_classifier=8, init_weights=True):
        super(VGG_Attention, self).__init__()
        self.num_classifier = num_classifier
        self.Embed_dim = Embed_dim
        self.features_0 = make_layers(cfg['C'], batch_norm=True)

        self.features_1 = make_layers(cfg['D'], batch_norm=True, in_channels=512)

        self.features_2 = make_layers(cfg['F'], batch_norm=True, in_channels=512)

        self.attention_blocks = nn.ModuleList([nn.Conv2d(
            512, 512, kernel_size=1, padding=0) for i in range(num_classifier)])

        self.Embedding = torch.nn.Sequential(
            torch.nn.Dropout(p=0.01),
            torch.nn.Linear(512, Embed_dim)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # soft_attentions = []

        x = self.features_0(x)
        y = self.features_1(x)
        soft_attentions = [nn.Sigmoid()(self.attention_blocks[i](y))
                           for i in range(self.num_classifier)]

        w = [torch.mul(x, soft_attentions[i])
             for i in range(self.num_classifier)]

        u = [self.features_2(self.features_1(w_)) for w_ in w]
        u = [u_.view(u_.size(0), -1) for u_ in u]

        u = [self.Embedding(u_) for u_ in u]
        u = torch.cat(u, 1)
        # u = u.view(x.size(0), -1)
        return u

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M7':
            layers += [nn.MaxPool2d(7)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [512, 512],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512],
    'F': ['M', 512, 512, 512, 'M', 'M7'],
}


def vgg_attention(**kwargs):
    """

    :param kwargs:
    :return: VGG attention model
    """

    model = VGG_Attention(**kwargs)
    return model
#
model = vgg_attention()
# print(model)
#
# model_dict = model.state_dict()
#
# dict_weight = [ v for k, v in model_dict.items() if k in model_dict]
#
#
# for w in dict_weight:
#     print(w.shape)
# #
# # print(dict_name)
# #
# pic = torch.ones(3, 3, 224, 224)
#
# out = model(pic)
#
# print(out[0].shape)

