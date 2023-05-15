import torchvision.models as models


def main():
  resnet18 = models.resnet18(pretrained=True)
  vgg16 = models.vgg16()


if __name__ == '__main__':
    main()
