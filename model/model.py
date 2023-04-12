import torch.nn as nn
import torch.nn.functional as F
import torch

class MaskModel(nn.Module):
    def __init__(self):
        super(MaskModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.mask_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

        self.gender_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        
        self.age_regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        mask = self.mask_classifier(x)
        gender = self.gender_classifier(x)
        age = self.age_regressor(x)
        return mask, gender, age
    

if __name__ == "__main__":
    model = MaskModel()
    x = torch.rand(1, 3, 512, 384)
    print(model(x).shape)