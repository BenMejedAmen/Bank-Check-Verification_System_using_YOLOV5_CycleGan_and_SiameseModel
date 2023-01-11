import os
import shutil
from io import BytesIO
from os import listdir
from os.path import isfile, join
from pathlib import Path
# import easyocr
import numpy
import torchvision
from django.shortcuts import render, redirect
import csv
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms

from Bank_chek_app.settings import MEDIA_ROOT, BASE_DIR
from .models import Cheque
from PIL import Image
import numpy as np
import cv2
import torch
import pandas as pd
import boto3
from PIL import Image
from django.forms.models import model_to_dict
from django.core.files.base import ContentFile, File
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import easyocr


ml_models_path = os.path.join(Path(__file__).resolve().parent, "ml_models")

yolov5_output = os.path.join(BASE_DIR, 'runs/detect')

model = torch.hub.load('ultralytics/yolov5', 'custom', path=ml_models_path + '/models/yolov5.pt', force_reload=True)


def process_image(img_path):
    img = cv2.imread(img_path)
    # img = np.array(img)
    # img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)

    # Normalisation
    normalisation_img = np.zeros((800, 800))
    normalisation_img = cv2.normalize(img, normalisation_img, 0, 255, cv2.NORM_MINMAX)

    # Grayscale
    grayscale_img = cv2.cvtColor(normalisation_img, cv2.COLOR_BGR2GRAY)

    # # Binary
    # ret, binary_img = cv2.threshold(grayscale_img, 127, 255, cv2.THRESH_BINARY)
    #
    # # Erotion
    # kernel = np.ones((1, 1), np.uint8)
    # erotion_img = cv2.erode(binary_img, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT)

    return grayscale_img


def put_bounding_box(img_path, image_object, image_name, bounding_box_info):
    # Here we will execute YOLOV5 model
    image = cv2.imread(img_path)
    result = image.copy()

    lower = np.array([90, 38, 0])
    upper = np.array([145, 255, 255])

    mask = cv2.inRange(image, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    result[close == 0] = (255, 255, 255)

    all_images = {}
    rows, columns = bounding_box_info.shape

    for row in range(0, rows):
        classe_info = bounding_box_info.iloc[row, -1]
        xmin = bounding_box_info.iloc[row, 0]
        ymin = bounding_box_info.iloc[row, 1]
        xmax = bounding_box_info.iloc[row, 2]
        ymax = bounding_box_info.iloc[row, 3]

        classe_info_image = image[int(ymin)-8: int(ymax)+8, int(xmin)-8: int(xmax)+8].copy()
        all_images[classe_info] = classe_info_image


    for key, image in all_images.items():
        path = MEDIA_ROOT + "\\" + "components\\" + key + "-" + image_name
        cv2.imwrite(path, image)

    images_directory = MEDIA_ROOT + "\\" + "components\\"
    if "issueBank" in all_images.keys():
        image_object.image_issueBank = images_directory + "issueBank-" + image_name
    if "ReceiverName" in all_images.keys():
        image_object.image_ReceiverName = images_directory + "ReceiverName-" + image_name
    if "amout_letter" in all_images.keys():
        image_object.image_amout_letter = images_directory + "amout_letter-" + image_name
    if "amount_digits" in all_images.keys():
        image_object.image_amount_digits = images_directory + "amount_digits-" + image_name
    if "cheque_number" in all_images.keys():
        image_object.image_cheque_number = images_directory + "cheque_number-" + image_name
    if "DateIss" in all_images.keys():
        image_object.image_DateIss = images_directory + "DateIss-" + image_name
    if "account_number" in all_images.keys():
        image_object.image_account_number = images_directory + "account_number-" + image_name
    if "signature" in all_images.keys():
        image_object.image_signature = images_directory + "signature-" + image_name
    image_object.save()


def ocr(image_path):
    #try:
    image_for_ocr = cv2.imread(image_path)
    success, encoded_image = cv2.imencode('.png', image_for_ocr)
    content2 = encoded_image.tobytes()

        #textractclient = boto3.client("textract", aws_access_key_id="#",aws_secret_access_key="#",region_name="us-east-2")

       # response = textractclient.detect_document_text(
       #     Document={
        #        'Bytes': content2
        #    }
       # )
    reader = easyocr.Reader(['en'], gpu=False)
    response =reader.readtext(image_for_ocr, detail=0, paragraph=True)
    extractedText = response

        #for block in response['Blocks']:
       #     if block["BlockType"] == "LINE":
       #         extractedText = extractedText + block["Text"] + " "

        # responseJson = {
        #     "text": extractedText
        # }
        # print(responseJson)
    print(extractedText)
    #except:
    #    return ""
    return extractedText


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 50, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # L1 ImgIn shape=(?, 28, 28, 1)      # (n-f+2*p/s)+1
        #    Conv     -> (?, 24, 24, 50)
        #    Pool     -> (?, 12, 12, 50)

        self.conv2 = nn.Conv2d(50, 60, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # L2 ImgIn shape=(?, 12, 12, 50)
        #    Conv      ->(?, 8, 8, 60)
        #    Pool      ->(?, 4, 4, 60)

        self.conv3 = nn.Conv2d(60, 80, kernel_size=3)
        # L3 ImgIn shape=(?, 4, 4, 60)
        #    Conv      ->(?, 2, 2, 80)

        self.batch_norm1 = nn.BatchNorm2d(50)
        self.batch_norm2 = nn.BatchNorm2d(60)

        #         self.dropout1 = nn.Dropout2d()

        # L4 FC 2*2*80 inputs -> 250 outputs
        self.fc1 = nn.Linear(32000, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward1(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        #         print(x.size())
        x = x.view(x.size()[0], -1)
        #         print('Output2')
        #         print(x.size()) #32000 thats why the input of fully connected layer is 32000
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward1(input1)
        # forward pass of input 2
        output2 = self.forward1(input2)

        return output1, output2


def test_similarity(dataset_image_path, uploaded_image_path):
    device = torch.device('cpu')
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(ml_models_path + '/models/siamese.pt', map_location='cpu'))

    trans = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

    img1 = Image.open(dataset_image_path)
    img11 = img1.convert('L')
    x0 = trans(img11)

    img2 = Image.open(uploaded_image_path)
    img22 = img2.convert('L')
    x1 = trans(img22)

    output1, output2 = model(x0.to(device).unsqueeze(1), x1.to(device).unsqueeze(1))

    eucledian_distance = F.pairwise_distance(output1, output2)

    print("Predicted Eucledian Distance:-", eucledian_distance)

    if eucledian_distance < 0.5:
        return 1
    else:
        return 0


def home(request):
    if request.method == 'POST':

        files = request.FILES


        new_image = Cheque()

        new_image.image_original = files["cheque_image"]
        new_image.save()

        image_original_path = new_image.image_original.path
        processed_image = process_image(image_original_path)
        processed_image_path = MEDIA_ROOT + "\\" + "preprocessing\\" + str(files["cheque_image"])
        cv2.imwrite(processed_image_path, processed_image)
        new_image.image_preprocessing = processed_image_path
        new_image.save()

        print("processed_image_path", processed_image_path)

        img = processed_image_path
        results = model(img)
        results.save()

        exp_directory = [f for f in listdir(yolov5_output) if not isfile(join(yolov5_output, f))][-1]
        image_filename = str(files["cheque_image"])

        yolov5_output_image_full_path = yolov5_output + "/" + exp_directory + "/" + image_filename
        shutil.copyfile(yolov5_output_image_full_path.replace("/","\\"), MEDIA_ROOT + "\\" + "yolov5\\" + image_filename)

        new_image.image_with_bounding_boxes = MEDIA_ROOT + "\\" + "yolov5\\" + image_filename
        new_image.save()

        put_bounding_box(processed_image_path, new_image, image_filename, results.pandas().xyxy[0])



        # window_name = 'image'
        # cv2.imshow(window_name, processed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # Redirect to Homepage
        return redirect('result')

    return render(request, "index.html")


def result(request):
    cheque = Cheque.objects.all().last()
    check_id_field = ocr(cheque.image_account_number)#.split(" ")
    final_id = check_id_field #[0]
   # for check_id in check_id_field:
   #     if len(check_id) > len(check_id):
    #        final_id = check_id

    check_nom = ocr(cheque.image_ReceiverName)
    check_cheque_number = ocr(cheque.image_cheque_number)
    check_date = ocr(cheque.image_DateIss)
    check_montant_chiffre = ocr(cheque.image_amount_digits)
    check_montant_lettre = ocr(cheque.image_amout_letter) #.replace("- U D'UN ORGANUME ASSIMILÉ 2 ", "")
    final_i = ''.join([str(item) for item in final_id])
    ocr_data = {
        'id': final_id,
        'nom': check_nom,
        'cheque_number': check_cheque_number,
        'date': check_date,
        'montant_chiffre': check_montant_chiffre,
        'montant_lettre': check_montant_lettre,
        'final_id': final_i,
    }

    id_from_image = cheque.image_original.path.split("\\")[-1].replace(".jpg", "")
    original_signature_img_path = MEDIA_ROOT + "\\" + "dataset\\" + final_i + ".jpg"
    uploaded_signature_img_path = MEDIA_ROOT + "\\" + "components\\" + "signature-" + id_from_image + ".jpg"

    result = test_similarity(original_signature_img_path, uploaded_signature_img_path)


    try:
        fields_length = [len(cheque.image_account_number), len(cheque.image_ReceiverName), len(cheque.image_cheque_number), len(cheque.image_DateIss),
                         len(cheque.image_signature), len(cheque.image_amount_digits), len(cheque.image_amout_letter)]
        error = 0
        for i in range(0, len(fields_length)):
            if (fields_length[i] == 0):
                error = 1
                break
    except:
        error = 1

    data = {
        "cheque": cheque,
        "error": error,
        "result": result,
        "ocr_data": ocr_data,
    }
    return render(request, "result.html", data)