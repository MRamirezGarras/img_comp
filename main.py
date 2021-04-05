import torch
from torchvision import transforms, models
import os
from PIL import Image
import pandas as pd
import numpy as np

#Directory names
inputDir = "images"
inputDirCNN = "transformed_images"

#Transform pictures to have same size
transformationForCNNInput = transforms.Compose([transforms.Resize((448,448))])
for imageName in os.listdir(inputDir):
    I = Image.open(os.path.join(inputDir, imageName))
    newI = transformationForCNNInput(I)

    if "exif" in I.info:
        exif = I.info['exif']
        newI.save(os.path.join(inputDirCNN, imageName), exif=exif)
    else:
        newI.save(os.path.join(inputDirCNN, imageName))


class Img2VecResnet34():
    def __init__(self):
        self.device = torch.device("cpu")
        self.numberFeatures = 512
        self.modelName = "resnet-34"
        self.model, self.featureLayer = self.getFeatureLayer()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def getFeatureLayer(self):
        cnnModel = models.resnet34(pretrained=True)
        layer = cnnModel._modules.get('avgpool')
        self.layer_output_size = 512

        return cnnModel, layer


    def getVec(self, img):
        image = self.normalize(self.toTensor(img)).unsqueeze(0).to(self.device)
        embedding = torch.zeros(1, self.numberFeatures, 1, 1)
        def copyData(m, i, o): embedding.copy_(o.data)
        h = self.featureLayer.register_forward_hook(copyData)
        self.model(image)
        h.remove()
        return embedding.numpy()[0, :, 0, 0]


img2vec = Img2VecResnet34()
allVectors = {}#Dictionary to store the values for each picture
for image in os.listdir(inputDirCNN):
    I = Image.open(os.path.join(inputDirCNN, image))
    vec = img2vec.getVec(I)
    allVectors[image] = vec
    I.close()


def getSimilarityMatrix(vectors):
    """Compares the pictures, gets a similarity score form 0 to 1"""
    v = np.array(list(vectors.values())).T
    sim = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    keys = list(vectors.keys())
    matrix = pd.DataFrame(sim, columns = keys, index = keys)
    return matrix

sim_matrix = getSimilarityMatrix(allVectors)

#Print the names of the pictures with a similarity higher than a certain threshold
threshold = 0.85

for col in sim_matrix.columns:
    data = sim_matrix[col]
    data = data[data > threshold]
    matches = [x for x in data.index if x != col]
    print ("Best matches", col, ":", matches)
