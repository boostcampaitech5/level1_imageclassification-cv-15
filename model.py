import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        self.model = models.wide_resnet101_2(pretrained=True)
        num_ftr = self.model.fc.in_features
        half_in_size = round(num_ftr/2)
        layer_width = 1024

        self.model.fc = SpinalNet(num_classes, half_in_size=half_in_size, layer_width=layer_width)

    def forward(self, x):

        x = self.model(x)
        return x

class ConvNext(nn.Module):
    def __init__(self, num_classes=18):
        super(ConvNext, self).__init__()
        self.base_model = timm.create_model('convnext_base', pretrained=True)  # pretrained convnext_base model을 사용
        self.base_model.head.fc = nn.Linear(self.base_model.head.fc.in_features, num_classes, bias=True)  # 마지막 레이어를 수정합니다.

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.densenet201(pretrained=True)
        # for name, param in self.model.named_parameters():
        #    param.requires_grad_(False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        # self.model.fc.requires_grad_(True)
    
    def forward(self, x):
        return self.model(x)
    
class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        # for name, param in self.model.named_parameters():
        #    param.requires_grad_(False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        # self.model.fc.requires_grad_(True)
    
    def forward(self, x):
        return self.model(x)    

class DenseNet201(nn.Module):
    def __init__(self, num_classes=18):
        super(DenseNet201, self).__init__()
        self.densenet_model = timm.create_model('densenet201', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        x = self.densenet_model(x)
        return x

class EfficientNetb0(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNetb0, self).__init__()
        self.efficientnet_model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.efficientnet_model(x)
        return x
    
if __name__ == "__main__":
    print("파라미터 갯수 출력")
    print("Resnet 101 Spinal FC : " , sum([param.numel() for param in ResNetSpinalFc(18).parameters() if param.requires_grad]))
    print("ResNext : " , sum([param.numel() for param in ResNext(18).parameters() if param.requires_grad]))
    print("Resnet34 : " , sum([param.numel() for param in Resnet34(18).parameters() if param.requires_grad]))
    print("MobileNetV2 : " , sum([param.numel() for param in MobileNetV2(18).parameters() if param.requires_grad]))
    print(DenseNet201(18))