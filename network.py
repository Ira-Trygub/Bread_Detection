from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.nn import ModuleDict
from collections import OrderedDict
from torchvision.models.detection.backbone_utils import BackboneWithFPN, IntermediateLayerGetter


#Make a  FasterRCNN Network with resnet50 backbone
def create_faster_rcnn_model(num_classes):
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model


def train_step(dataloader, device, model, optimizer,  total_loss, lr_scheduler, epoch, output_path ):
        model.train()
        print("Wir are in train_step, epoch: ", epoch)
        for img_ids, images, boxes, classes in dataloader:
            images = [image.to(device) for image in images]
            targets = []
            for b, c in zip(boxes, classes):
                b = torch.as_tensor(b, dtype=torch.float32).to(device) 
                c = torch.as_tensor(c, dtype=torch.int64).to(device)  
                targets.append({'boxes': b, 'labels': c})

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}, output_path)
        return total_loss / len(dataloader)





