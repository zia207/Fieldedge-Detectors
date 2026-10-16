"""Registry of the converted segmentation models."""

from __future__ import annotations

from models.adapnet import AdapNet
from models.deeplab import DeepLabV3, DeepLabV3Plus
from models.encoder_decoder import EncoderDecoder
from models.fc_densenet import FCDenseNet
from models.frrn import FRRN
from models.gcn import GCN
from models.icnet import ICNet
from models.mobile_unet import MobileUNet
from models.pspnet import PSPNet
from models.refinenet import RefineNet

# Names match the presets in cropboundarycode/main.py, plus the ICNet presets
# defined in cropboundarycode/models/ICNet.py.
MODEL_FACTORIES = {
    "FC-DenseNet56": lambda num_classes: FCDenseNet("FC-DenseNet56", num_classes),
    "FC-DenseNet67": lambda num_classes: FCDenseNet("FC-DenseNet67", num_classes),
    "FC-DenseNet103": lambda num_classes: FCDenseNet("FC-DenseNet103", num_classes),
    "Encoder-Decoder": lambda num_classes: EncoderDecoder("Encoder-Decoder", num_classes),
    "Encoder-Decoder-Skip": lambda num_classes: EncoderDecoder("Encoder-Decoder-Skip", num_classes),
    "FRRN-A": lambda num_classes: FRRN("FRRN-A", num_classes),
    "FRRN-B": lambda num_classes: FRRN("FRRN-B", num_classes),
    "MobileUNet": lambda num_classes: MobileUNet("MobileUNet", num_classes),
    "MobileUNet-Skip": lambda num_classes: MobileUNet("MobileUNet-Skip", num_classes),
    "AdapNet": lambda num_classes: AdapNet(num_classes),
    "RefineNet-Res50": lambda num_classes: RefineNet("RefineNet-Res50", num_classes),
    "RefineNet-Res101": lambda num_classes: RefineNet("RefineNet-Res101", num_classes),
    "RefineNet-Res152": lambda num_classes: RefineNet("RefineNet-Res152", num_classes),
    "PSPNet-Res50": lambda num_classes: PSPNet("PSPNet-Res50", num_classes),
    "PSPNet-Res101": lambda num_classes: PSPNet("PSPNet-Res101", num_classes),
    "PSPNet-Res152": lambda num_classes: PSPNet("PSPNet-Res152", num_classes),
    "GCN-Res50": lambda num_classes: GCN("GCN-Res50", num_classes),
    "GCN-Res101": lambda num_classes: GCN("GCN-Res101", num_classes),
    "GCN-Res152": lambda num_classes: GCN("GCN-Res152", num_classes),
    "DeepLabV3-Res50": lambda num_classes: DeepLabV3("DeepLabV3-Res50", num_classes),
    "DeepLabV3-Res101": lambda num_classes: DeepLabV3("DeepLabV3-Res101", num_classes),
    "DeepLabV3-Res152": lambda num_classes: DeepLabV3("DeepLabV3-Res152", num_classes),
    "DeepLabV3_plus-Res50": lambda num_classes: DeepLabV3Plus("DeepLabV3_plus-Res50", num_classes),
    "DeepLabV3_plus-Res101": lambda num_classes: DeepLabV3Plus("DeepLabV3_plus-Res101", num_classes),
    "DeepLabV3_plus-Res152": lambda num_classes: DeepLabV3Plus("DeepLabV3_plus-Res152", num_classes),
    "ICNet-Res50": lambda num_classes: ICNet("ICNet-Res50", num_classes),
    "ICNet-Res101": lambda num_classes: ICNet("ICNet-Res101", num_classes),
    "ICNet-Res152": lambda num_classes: ICNet("ICNet-Res152", num_classes),
}
