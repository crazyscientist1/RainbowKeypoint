from environment import Variables
from transform_toolkit import augmenter, draw_labelmap
import zipfile
import os
import json
from PIL import Image
import numpy as np
import napari


class DataHandler():
    def unzip(self):
        if not os.path.isdir(Variables.BASE_PATH + 'dataset/content'):
            with zipfile.ZipFile(Variables.BASE_PATH + 'dataset/images.zip', 'r') as zip_ref:
                zip_ref.extractall(Variables.BASE_PATH + 'dataset')

    def unpack(self):
        #Poorly written function to
        images = [[], [], [], [], [], [], []]
        labels = [[], [], [], [], [], [], []]

        f = open(Variables.BASE_PATH + 'dataset/content/images/annotations.json')
        data = json.load(f)

        for i in data['metadata']:
            fileName = data['file'][data['metadata'][i]['vid']]['fname']

            image = Image.open(Variables.BASE_PATH + 'dataset/content/images/' + fileName)
            xCoord = int(data['metadata'][i]['xy'][1])
            yCoord = int(data['metadata'][i]['xy'][2])

            xScale = xCoord / np.shape(image)[1]
            yScale = yCoord / np.shape(image)[0]


            image = image.resize(Variables.INP_SIZE)

            images[int(data['metadata'][i]['vid']) - 1] = image
            labels[int(data['metadata'][i]['vid']) - 1].append([xScale * Variables.INP_SIZE[0], yScale * Variables.INP_SIZE[1]])

        images.pop(5)
        labels.pop(5)

        f = open(Variables.BASE_PATH + 'dataset/content/images/rainbowProject1.json')
        data = json.load(f)

        for i in data['_via_img_metadata']:

            fileName = data['_via_img_metadata'][i]['filename']

            image = Image.open(Variables.BASE_PATH + 'dataset/content/images/' + fileName)

            labels.append([])
            for x in data['_via_img_metadata'][i]['regions']:
                xCoord, yCoord = int(x['shape_attributes']['cy']), int(x['shape_attributes']['cx'])
                xScale = yCoord / np.shape(image)[1]
                yScale = xCoord / np.shape(image)[0]

                labels[len(labels) - 1].append([xScale * Variables.INP_SIZE[0], yScale * Variables.INP_SIZE[1]])

            images.append(image)
            image = image.resize(Variables.INP_SIZE)

        return images, labels
    def augmentEpoch(self, epochSize):
        max_length = max(len(sublist) for sublist in self.labels)

        outImg = np.empty( (0,) + np.shape(self.images[0]))
        outLabels = np.empty( (0,) + (max_length,) + Variables.OUT_SIZE )



        for i in range(epochSize):

            img, label = augmenter(self.images[i % len(self.images)], Variables.INP_SIZE[0], self.labels[i % len(self.images)])

            outImg = np.vstack((outImg, np.expand_dims(img, axis=0)))

            labelGrid = np.zeros((max_length,) + Variables.OUT_SIZE)
            for i, coords in enumerate(label):
                out = np.array(coords) * Variables.OUT_SIZE[0]
                labelGrid[i] = draw_labelmap(labelGrid[i], out.astype(int), sigma = Variables.SIGMA)

            outLabels = np.vstack((outLabels, np.expand_dims(labelGrid, axis = 0)))


        return outImg, outLabels

    def __init__(self):
        self.unzip()

        self.images, self.labels = self.unpack()
