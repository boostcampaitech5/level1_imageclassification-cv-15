import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch
import timm

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

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

class SpinalNet(nn.Module):
    def __init__(self, num_classes, half_in_size, layer_width):
        super(SpinalNet, self).__init__()
        self.half_in_size = half_in_size
        self.layer_width = layer_width

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_in_size, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(self.half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = 0.5), nn.Linear(layer_width*4, num_classes),)
        
    def forward(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:self.half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:self.half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x3], dim=1))
                
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)
        return x

# Custom Model Template
class ResNetSpinalFc(nn.Module):
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
        x = self.base_model(x)
        return x
    
class ConvNext_22k(nn.Module):
    def __init__(self, num_classes=18):
        super(ConvNext_22k, self).__init__()
        self.base_model = timm.create_model('convnext_base_in22k', pretrained=True)  # pretrained convnextv2_base model을 사용
        self.base_model.head.fc = nn.Linear(self.base_model.head.fc.in_features, num_classes, bias=True)  # 마지막 레이어를 수정합니다.

    def forward(self, x):
        x = self.base_model(x)
        return x

class Resnet18(nn.Module):
    def __init__(self, num_classes=18):
        super(Resnet18, self).__init__()
        self.base_model = models.resnet18(pretrained=True)  # pretrained resnet18 model을 사용
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes, bias=True)  # 마지막 레이어를 수정합니다.

    def forward(self, x):
        x = self.base_model(x)
        return 

class ResNext101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnext101_32x8d(pretrained=True)
        # for name, param in self.model.named_parameters():
        #    param.requires_grad_(False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # self.model.fc.requires_grad_(True)
    
    def forward(self, x):
        return self.model(x)

class ResNext50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnext50_32x4d(pretrained=True)
        # for name, param in self.model.named_parameters():
        #    param.requires_grad_(False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # self.model.fc.requires_grad_(True)
    
    def forward(self, x):
        return self.model(x)

class Resnet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet34(pretrained=True)
        # for name, param in self.model.named_parameters():
        #    param.requires_grad_(False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        # self.model.fc.requires_grad_(True)
    
    def forward(self, x):
        return self.model(x)
    
class DenseNet201(nn.Module):
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
        #self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
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

class EfficientNetb2(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNetb2, self).__init__()
        self.efficientnet_model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes, drop_rate=0.2)
    
    def forward(self, x):
        x = self.efficientnet_model(x)
        return x

class EfficientNetb3(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNetb3, self).__init__()
        self.efficientnet_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes, drop_rate=0.2)
    
    def forward(self, x):
        x = self.efficientnet_model(x)
        return x
    

class EfficientNetb4(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNetb4, self).__init__()
        self.efficientnet_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.efficientnet_model(x)
        return x

class EfficientNetV2_M(nn.Module):
    def __init__(self, num_classes=18):
        super(EfficientNetV2_M, self).__init__()
        self.efficientnet_model = timm.create_model('efficientnetv2_m', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.efficientnet_model(x)
        return x


class VitB(nn.Module):
    def __init__(self, num_classes=18):
        super(VitB, self).__init__()
        self.vit_model = timm.models.vit_base_patch16_224(pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        x = self.vit_model(x)
        return x
        
if __name__ == "__main__":
    print("파라미터 갯수 출력")
    print("Resnet 101 Spinal FC : " , sum([param.numel() for param in ResNetSpinalFc(18).parameters() if param.requires_grad]))
    print("ResNext : " , sum([param.numel() for param in ResNext101(18).parameters() if param.requires_grad]))
    print("Resnet34 : " , sum([param.numel() for param in Resnet34(18).parameters() if param.requires_grad]))
    print("MobileNetV2 : " , sum([param.numel() for param in MobileNetV2(18).parameters() if param.requires_grad]))
    print(DenseNet201(18))
