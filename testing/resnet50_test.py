import os
import cv2
import numpy as np
import tensorflow as tf
from Retinanet.models.resnet_50 import Resnet50
from Retinanet.preprocessing.resnet_dg import ResnetDataGenerator

def load_data_files(test_dir , num_samples):
    files = os.listdir(test_dir)
    np.random.shuffle(files)
    return files[0:num_samples]

def load_images(sample_files , data_path , norm):
    ec_images = []
    org_images = []
    for sample in sample_files:
        file_path = os.path.join(data_path, sample)
        image = cv2.cvtColor(cv2.imread(file_path) , cv2.COLOR_BGR2RGB)
        image = cv2.resize(image , (224 , 224))
        org_images.append(image)
        if norm:
            image = image / 255.
        ec_images.append(image)

    return org_images , ec_images

def load_model_weights(model , chk_dir):
    ckpt = tf.train.get_checkpoint_state(chk_dir)
    if ckpt and ckpt.model_checkpoint_path:
        model.load_weights(ckpt.model_checkpoint_path)
        print("pretrained model loaded")

    else:
        raise Exception("No pretrained model found.")
    return model

def decode_preds(preds , org_samples , num_samples):
    for i in range(num_samples):
        img = org_samples[i]
        cv2.imshow("img" , img[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(preds[i])

def test_resnet50():
    num_samples = 10
    resnet50 = Resnet50(224 , include_top = True , num_classes = 2)
    test_dir = "/home/yogeesh/yogeesh/datasets/car/data/testing_images"
    chk_dir = "../saved_model/resnet50/"
    sample_files = load_data_files(test_dir , num_samples)
    org_samples , ec_samples = load_images(sample_files , test_dir , True)
    ec_samples = np.reshape(ec_samples , (num_samples , 224 , 224, 3))
    resnet50 = load_model_weights(resnet50 , chk_dir)
    preds = resnet50(ec_samples , training = False)
    decode_preds(preds , org_samples , num_samples)

if __name__ == "__main__":
    test_resnet50()
