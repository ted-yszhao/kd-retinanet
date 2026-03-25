from collections import OrderedDict


def extract_fpn_feats(model, images):
    """
    Extract FPN features from the model.
    Args:
        model: The RetinaNet model.
        images: A list of input images (tensors).
    Returns:
        A dictionary of FPN features.
    """
    image_list, _ = model.transform(images, None)
    feats = model.backbone(image_list.tensors)
    if not isinstance(feats, OrderedDict):
        feats = OrderedDict({"0": feats})
    return feats