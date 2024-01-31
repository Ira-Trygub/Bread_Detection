import torch
from util import calculate_iou
import  cv2
import numpy as np


def test_model(dataloader, model, device, iou_threshold=0.5):
    model.eval()
    with torch.no_grad():
        all_detections = []
        all_annotations = []
        vals = []
        model = model.to(device)
        scores = []

        for img_ids, images, gt_boxes, gt_classes in dataloader:
            images = [image.to(device) for image in images]
            predictions = model(images)
            
            for i, prediction in enumerate(predictions):
                detections = []
                annotations = []
                
                pred_boxes = prediction['boxes'].cpu().numpy()
                pred_scores = prediction['scores'].cpu().numpy()
                pred_labels = prediction['labels'].cpu().numpy()

                gt_boxes_i = gt_boxes[i].cpu().numpy()
                gt_classes_i = gt_classes[i].cpu().numpy()

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if score >= iou_threshold:
                        detections.append([box, score, label])
                        scores.append(score)
                       

                for box, label in zip(gt_boxes_i, gt_classes_i):
                    annotations.append([box, label])

                #print("pred_boxes, gt_boxes_i ", pred_boxes, gt_boxes_i)
                iou = 0
                for pr_box, gt_box in zip(pred_boxes, gt_boxes_i):
                    iou = calculate_iou(pr_box, gt_box)
                    vals.append(iou[0])

                all_detections.append(detections)
                all_annotations.append(annotations)

        mean_score = sum(scores)/(len(scores)+1)
    return  mean_score 

