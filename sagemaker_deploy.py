from sagemaker.pytorch import PyTorchModel
import sagemaker
from time import gmtime, strftime
import boto3
from io import BytesIO
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances, Boxes

import torch
import cv2
import pickle
import os
# consts
sess = sagemaker.Session()
role = sagemaker.get_execution_role()
account = sess.boto_session.client('sts').get_caller_identity()['Account']

region = 'us-east-1'
n = 'deploy'
model_name = f"d2-{n}"
endpoint_name = f"d2-{n}"

# Update this with the model output location `model.tar.gz` file
model_url =  r"s3://sagemaker-blueberry-test/test_model.tar.gz" # Should look like s3://PATH_TO_OUTPUT/model.tar.gz

remote_model = PyTorchModel(
                     name = model_name,
                     model_data=model_url,
                     role=role,
                     sagemaker_session = sess,
                     entry_point="inference.py",
                     # image=image,
                     framework_version="1.6.0",
                     py_version='py3'
                    )

remote_predictor = remote_model.deploy(
                         instance_type='ml.g4dn.xlarge',
                         initial_instance_count=1,
                         # update_endpoint = True, # comment or False if endpoint doesns't exist
                         endpoint_name=endpoint_name, # define a unique endpoint name; if ommited, Sagemaker will generate it based on used container
                         wait=True
                         )

client = boto3.client('sagemaker-runtime')

accept_type = "json"
content_type = 'image/jpeg'
headers = {'content-type': content_type}
device = "cpu" # cuda:0 for GPU, cpu for CPU
test_pics_dir = '<PATH_TO_SOME_TEST_IMGS>'

classID_name = {
    1: 'Blueberry'
}

def json_to_d2(pred_dict, device):
    """
    Client side helper function to deserialize the JSON msg back to d2 outputs
    """

    # pred_dict = json.loads(predictions)
    for k, v in pred_dict.items():
        if k=="pred_boxes":
            boxes_to_tensor = torch.FloatTensor(v).to(device)
            pred_dict[k] = Boxes(boxes_to_tensor)
        if k=="scores":
            pred_dict[k] = torch.Tensor(v).to(device)
        if k=="pred_classes":
            pred_dict[k] = torch.Tensor(v).to(device).to(torch.uint8)

    height, width = pred_dict['image_size']
    del pred_dict['image_size']

    inst = Instances((height, width,), **pred_dict)

    return {'instances':inst}

for img_ in os.listdir(test_pics_dir):

    img_name = test_pics_dir + img_
    print(img_name)

    payload = open(img_name, 'rb')
    device = "cpu"

    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload,
        ContentType=content_type,
        Accept = accept_type
    )

    if accept_type=="json":
        predictions = json_to_d2(response['Body'].read(), device)
    elif accept_type=="detectron2":
        print(response['Body'].read())
        stream = BytesIO(response['Body'].read())
        predictions = pickle.loads(stream.read())

    # Extract preds:
    preds = predictions["instances"].to("cpu")
    boxes = preds.pred_boxes.tensor.numpy()
    scores = preds.scores.tolist()
    classes = preds.pred_classes.tolist()

    for i in range(len(boxes)):
        left, top, right, bot = boxes[i] #x0, y0, x1, y1
        print(f'DETECTED: {classID_name[classes[i]]}, confidence: {scores[i]}, box: {int(left)} {int(top)} {int(right)} {int(bot)}\n') # left top right bot

    # visualize
    im = cv2.imread(img_name)
    v = Visualizer(im[:, :, ::-1],
                   metadata="train", # Extracted from Visualizer.get() code
                   scale=0.8)
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    cv2.imshow(out.get_image()[:, :, ::-1])


