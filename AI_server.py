#HTTP Server for communication with Java client
from flask import Flask, request, jsonify
import torch
import io
from PIL import Image
from network import create_faster_rcnn_model
import base64 
from torchvision import transforms
import yaml

ai_server = Flask(__name__)

def load_model():
    device = torch.device('cpu')
    with open('cfg.yaml', 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)

    num_classes = cfg["model"]["num_classes"]  
    output_path = "D:/HAW-Hamburg_E/VS/Bread_Detection/output/model_2023_12_13_2.pth"

       
    saved_dict = torch.load(output_path , map_location=device)
    
   
    model_state_dict = saved_dict['model']
    model = create_faster_rcnn_model(num_classes)
   
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

@ai_server.route('/scanPhoto', methods=['POST'])
def scanPhoto():
    print("I am living jet 35")
    data = request.json
    print("json: ", data)
    image_string = data['image']#.encode("ascii", "utf-8")
    image_bytes = base64.b64decode(image_string)
    image_bytes_io = io.BytesIO(image_bytes)     
    image = Image.open(image_bytes_io) 
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    tensor_image = transform(image)
    
    model = load_model()

    with torch.no_grad():
        prediction = model([tensor_image])

    print("prediction: ", type(prediction), prediction)

    prediction_list = []
    for element in prediction:
        # Преобразование тензоров в списки
        boxes = element['boxes'].tolist()
        labels = element['labels'].tolist()
        scores = element['scores'].tolist()

        prediction_list.append({
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        })

  
    return jsonify(prediction_list)
 


@ai_server.route('/print', methods=['GET'])
def printHalloWorld():
    return "Hallo World!!!"


if __name__=='__main__':
    ai_server.run(debug = True, port = 5003)