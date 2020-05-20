import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self, image_size=224, num_class=2,
                 fea_ext=models.vgg19_bn, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.image_size = image_size
        # for binary classification
        self.num_class = num_class
        self.feature_extractor = fea_ext(pretrained=pretrained)
        if isinstance(self.feature_extractor, models.ResNet):
            self.in_fea = self.feature_extractor.fc.in_features
            # drop last Linear layer
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-1]
            )
        elif isinstance(self.feature_extractor, models.VGG):
            self.in_fea = 512 * 7 * 7
            # drop classifier
            self.feature_extractor = nn.Sequential(
                self.feature_extractor.features,
                self.feature_extractor.avgpool,
            )
        else:
            raise NotImplementedError("{} is not supported." % fea_ext)

    def forward(self, images):
        fea = self.feature_extractor(images)
        return fea
