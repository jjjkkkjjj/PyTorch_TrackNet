from torch import nn

class Conv2dRelu:
    batch_norm = True

    @staticmethod
    def block(order, block_num, in_channels, out_channels, **kwargs):
        """
        :param order: int or str
        :param block_num: int, how many conv layers are sequenced
        :param in_channels: int
        :param out_channels: int
        :param batch_norm: bool
        :param kwargs:
        :return: list of tuple is for OrderedDict
        """
        kernel_size = kwargs.pop('conv_k_size', (3, 3))
        stride = kwargs.pop('conv_stride', (1, 1))
        padding = kwargs.pop('conv_padding', 1)
        relu_inplace = kwargs.pop('relu_inplace', False)# TODO relu inplace problem >>conv4
        batch_norm = kwargs.pop('batch_norm', Conv2dRelu.batch_norm)

        in_c = in_channels
        layers = []
        # append conv block
        for bnum in range(block_num):
            postfix = '{0}_{1}'.format(order, bnum + 1)
            if not batch_norm:
                layers += [
                    ('conv{}'.format(postfix),
                     nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                    ('relu{}'.format(postfix), nn.ReLU(relu_inplace))
                ]
            else:
                layers += [
                    ('conv{}'.format(postfix),
                     nn.Conv2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                    ('bn{}'.format(postfix), nn.BatchNorm2d(out_channels)),
                    ('relu{}'.format(postfix), nn.ReLU(relu_inplace))
                ]
            in_c = out_channels

        kernel_size = kwargs.pop('pool_k_size', (2, 2))
        stride = kwargs.pop('pool_stride', (2, 2))
        ceil_mode = kwargs.pop('pool_ceil_mode', False)
        padding = kwargs.pop('pool_padding', 0)
        # append maxpooling
        layers += [
            ('pool{}'.format(order), nn.MaxPool2d(kernel_size, stride=stride, ceil_mode=ceil_mode, padding=padding))
        ]

        return layers

    @staticmethod
    def one(postfix, *args, relu_inplace=False, **kwargs):
        batch_norm = kwargs.pop('batch_norm', Conv2dRelu.batch_norm)
        if not batch_norm:
            return [
                ('conv{}'.format(postfix), nn.Conv2d(*args, **kwargs)),
                ('relu{}'.format(postfix), nn.ReLU(inplace=relu_inplace))
            ]
        else:
            out_channels = kwargs.pop('out_channels', args[1])
            return [
                ('conv{}'.format(postfix), nn.Conv2d(*args, **kwargs)),
                ('bn{}'.format(postfix), nn.BatchNorm2d(out_channels)),
                ('relu{}'.format(postfix), nn.ReLU(inplace=relu_inplace))
            ]

class Deconv2dRelu:
    batch_norm = True

    @staticmethod
    def block(order, block_num, in_channels, out_channels, **kwargs):
        """
        :param order: int or str
        :param block_num: int, how many deconv layers are sequenced
        :param in_channels: int
        :param out_channels: int
        :param batch_norm: bool
        :param kwargs:
        :return: list of tuple is for OrderedDict
        """
        layers = []

        kernel_size = kwargs.pop('upsample_k_size', (2, 2))
        scale_factor = kwargs.pop('scale_factor', None)
        # append upsampling
        layers += [
            ('upsample{}'.format(order), nn.UpsamplingNearest2d(kernel_size, scale_factor=scale_factor))
        ]

        kernel_size = kwargs.pop('deconv_k_size', (3, 3))
        stride = kwargs.pop('deconv_stride', (1, 1))
        padding = kwargs.pop('deconv_padding', 1)
        relu_inplace = kwargs.pop('relu_inplace', True)# TODO relu inplace problem >>conv4
        batch_norm = kwargs.pop('batch_norm', Deconv2dRelu.batch_norm)

        in_c = in_channels
        # append deconv block
        for bnum in range(block_num):
            postfix = '{0}_{1}'.format(order, bnum + 1)
            if not batch_norm:
                layers += [
                    ('deconv{}'.format(postfix),
                     nn.ConvTranspose2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                    ('relu{}'.format(postfix), nn.ReLU(relu_inplace))
                ]
            else:
                layers += [
                    ('deconv{}'.format(postfix),
                     nn.ConvTranspose2d(in_c, out_channels, kernel_size, stride=stride, padding=padding)),
                    ('bn{}'.format(postfix), nn.BatchNorm2d(out_channels)),
                    ('relu{}'.format(postfix), nn.ReLU(relu_inplace))
                ]
            in_c = out_channels

        return layers

    @staticmethod
    def one(postfix, *args, relu_inplace=True, **kwargs):
        batch_norm = kwargs.pop('batch_norm', Deconv2dRelu.batch_norm)
        if not batch_norm:
            return [
                ('deconv{}'.format(postfix), nn.ConvTranspose2d(*args, **kwargs)),
                ('relu{}'.format(postfix), nn.ReLU(inplace=relu_inplace))
            ]
        else:
            out_channels = kwargs.pop('out_channels', args[1])
            return [
                ('deconv{}'.format(postfix), nn.ConvTranspose2d(*args, **kwargs)),
                ('bn{}'.format(postfix), nn.BatchNorm2d(out_channels)),
                ('relu{}'.format(postfix), nn.ReLU(inplace=relu_inplace))
            ]