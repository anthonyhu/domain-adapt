import math
import torchvision.transforms as transforms
import torch.nn.init as init

def convert_to_pil(x):
    x = x*0.5 + 0.5
    return transforms.ToPILImage()(x)


def weights_init(init_type='gaussian'):
    def init_fun(m):
        class_name = m.__class__.__name__
        if (class_name.find('Conv') == 0 or class_name.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun