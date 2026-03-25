from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.detection import RetinaNet, retinanet_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def build_teacher(num_classes: int = 91):
    """
    
    Build the teacher RetinaNet model with ResNet101 backbone.
    Args:
        num_classes: Number of classes for detection (including background).
    Returns:
        A RetinaNet model with ResNet101 backbone.
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.DEFAULT,
        trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    return RetinaNet(backbone=backbone, num_classes=num_classes)


def build_student(num_classes: int = 91):
    return retinanet_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
        trainable_backbone_layers=3,
    )