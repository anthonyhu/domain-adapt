import torchvision.transforms as transforms


def convert_to_pil(x):
    x = x*0.5 + 0.5
    return transforms.ToPILImage()(x)