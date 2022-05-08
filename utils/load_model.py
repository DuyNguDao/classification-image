import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import pickle
import cv2


class Model:
    def __init__(self, path):
        # config device cuda or cpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model = mobilenet_v2(pretrained=False, num_classes=2).to(self.device)
        self.path = path
        self.load_model()
        self.input_size = 32


    def load_model(self):
        """
        function: load model and parameter
        :return:
        """
        # load model
        with open(self.path + '/model.pickle', 'rb') as file:
            self.model = pickle.load(file)

        self.model.load_state_dict(torch.load(self.path + "/best.h5", map_location=self.device))
        # load size of input
        with open(self.path + '/parameter.txt', 'r') as file:
            data = file.readline()
            data = data.split('=')[1].strip()
            self.input_size = int(data)
            file.close()
        # load label and id
        with open(self.path + '/label.txt', 'r') as file:
            data = file.readlines()
            file.close()

        self.class_name = []
        self.class_id = []
        for i in data:
            name = i.split(':')
            self.class_name.append(name[0].strip())
            self.class_id.append(int(name[1].strip()))
        self.class_name_id = zip(self.class_name, self.class_id)
        self.class_name_id = dict(self.class_name_id)

    def preprocess_image(self, image):
        """
        function: preprocessing image
        :param image: array image
        :return:
        """
        # convert BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        input_size = self.input_size
        # normalize with imagenet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # preprocess image
        preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize])
        img_preprocessed = preprocess(img)
        batch_img_tensor = torch.unsqueeze(img_preprocessed, 0)
        return batch_img_tensor

    def predict(self, image):
        """
        function: predict image
        :param image: array image bgr
        :return: name class predict and list prob predict
        """
        img = self.preprocess_image(image).to(self.device)
        self.model.eval()
        # predict
        out = self.model(img)
        # find max
        _, index = torch.max(out, 1)
        # find prob use activation softmax
        percentage = (nn.functional.softmax(out, dim=1)[0] * 100).tolist()
        return self.class_name[index], percentage
