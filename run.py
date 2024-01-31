#Recognition of bakery products for the 'Digital Bakery' project
import  PIL
import yaml
import torch
from Bread_Dataloader  import  BreadDataset
from torch.utils.data import DataLoader
from network import create_faster_rcnn_model, train_step 
import argparse
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from test import  test_model

from torch.utils.tensorboard import SummaryWriter


#This function is necessary to feed images of different sizes into the model
def custom_collate_fn(batch):
        img_id =  [item[0] for item in batch]
        images = [item[1] for item in batch]  
        boxes = [item[2] for item in batch]  
        classes = [item[3] for item in batch] 
        images = torch.stack(images, dim=0)
        return img_id, images, boxes, classes

def load_model(model, device, num_classes):
    model.load_state_dict(torch.load(model, map_location=device))
    return model

def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
         device = torch.device('cpu')
    print("device: ", device)
    print(torch.cuda.is_available())
    with open('cfg.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    output_path = cfg["data"]["output"]
    num_classes = cfg["model"]["num_classes"]  
    writer = SummaryWriter(cfg["training"]["tb"])

    batch_size=cfg["training"]["batch_size"]
    dataset = BreadDataset("cfg.yaml")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,  collate_fn=custom_collate_fn)

    test_dataset = BreadDataset("cfg.yaml", train = False)
    test_dataloader= DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2,  collate_fn=custom_collate_fn)

    
    model = create_faster_rcnn_model(num_classes)
    if args.test:
        saved_dict = torch.load(output_path , map_location=device)
        model_state_dict = saved_dict['model']
        model.load_state_dict(model_state_dict)
        for parameter in model.parameters():
             parameter = parameter.to(device)

        score = test_model(test_dataloader, model, device, iou_threshold=0.7)
        writer.add_scalar('score', score, epoch)
        return score
        
 
    else:
        model.train() 
        total_loss = 0
        model.to(device)
        num_epochs = cfg["training"]["num_epochs"]
        lr= cfg["training"]["learning_rate"]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        for epoch in range(num_epochs):
            epoch_loss = train_step(dataloader, device, model, optimizer,  total_loss, lrscheduler, epoch , output_path)
            print(f"Epoch {epoch+1}, Loss: {epoch_loss}")
            score = test_model(dataloader, model, device, iou_threshold=0.5)
            writer.add_scalar('score', score, epoch)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Detection")
    parser.add_argument("-test", action="store_true") 
    args = parser.parse_args()
    main(args)
