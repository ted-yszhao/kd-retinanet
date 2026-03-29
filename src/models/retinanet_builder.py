from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.detection import RetinaNet, retinanet_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


def _resolve_resnet101_weights(name):
    if name is None:
        return None
    return getattr(ResNet101_Weights, name)


def _resolve_resnet50_weights(name):
    if name is None:
        return None
    return getattr(ResNet50_Weights, name)


def build_teacher(
    num_classes: int = 91,
    backbone_name: str = "resnet101",
    backbone_weights="DEFAULT",
    trainable_layers: int = 3,
    returned_layers=None,
    extra_blocks_in_channels: int = 256,
    extra_blocks_out_channels: int = 256,
):
    """
    
    Build the teacher RetinaNet model with ResNet101 backbone.
    Args:
        num_classes: Number of classes for detection (including background).
    Returns:
        A RetinaNet model with ResNet101 backbone.
    """
    if returned_layers is None:
        returned_layers = [2, 3, 4]

    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=_resolve_resnet101_weights(backbone_weights),
        trainable_layers=trainable_layers,
        returned_layers=returned_layers,
        extra_blocks=LastLevelP6P7(extra_blocks_in_channels, extra_blocks_out_channels),
    )
    return RetinaNet(backbone=backbone, num_classes=num_classes)


def build_student(
    num_classes: int = 91,
    detector_weights=None,
    backbone_weights="DEFAULT",
    trainable_backbone_layers: int = 3,
):
    return retinanet_resnet50_fpn(
        weights=detector_weights,
        weights_backbone=_resolve_resnet50_weights(backbone_weights),
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
    )
