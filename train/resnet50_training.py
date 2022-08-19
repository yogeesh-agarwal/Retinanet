import cv2
import numpy as np
import tensorflow as tf
from Retinanet.models.resnet_50 import Resnet50
from Retinanet.preprocessing.resnet_dg import ResnetDataGenerator

class ResnetTrainer():
    def __init__(self , data_path, true_csv_path , lr , bs , num_classes ,
                 norm = True , shuffle = True, inp_size = 224, num_epochs = 1):
        self.norm = norm
        self.shuffle = shuffle
        self.batch_size = bs
        self.init_lr = lr
        self.input_size = inp_size
        self.num_epochs = num_epochs
        self.data_path = data_path
        self.num_classes = num_classes
        self.true_csv_path = true_csv_path
        self.log_dir = "../logs/"
        self.save_dir = "../saved_model/resnet50/"

        self.train_data_generator = ResnetDataGenerator(self.true_csv_path ,
                                                        self.data_path ,
                                                        self.batch_size ,
                                                        self.input_size ,
                                                        self.input_size ,
                                                        num_classes = self.num_classes,
                                                        shuffle = self.shuffle,
                                                        norm = self.norm)

        self.resnet50 = Resnet50(self.input_size , include_top = True ,num_classes = self.num_classes)

    def gen_callbacks(self , tb_ld , cp_path):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tb_ld,
                                                              histogram_freq = 1,
                                                              update_freq = "batch")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path,
                                                                 verbose = 1,
                                                                 mode = "min",
                                                                 save_best_only = True,
                                                                 monitor = "train_loss",
                                                                 save_weights_only = True)
        reduce_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor = "train_loss",
                                                                          factor = 0.5,
                                                                          patience = 20,
                                                                          verbose = 1,
                                                                          mode = "min",
                                                                          min_lr = 1e-6)
        return [tensorboard_callback , checkpoint_callback , reduce_on_plateau_callback]

    def train(self):
        callbacks = self.gen_callbacks(self.log_dir , self.save_dir+"resnet50.ckpt")
        self.resnet50.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.init_lr))
        history = self.resnet50.fit(self.train_data_generator , epochs = self.num_epochs ,
                          use_multiprocessing = True , initial_epoch = 0 , callbacks = callbacks)

        print("training Completed")

if __name__ == "__main__":
    true_csv_path = "/home/yogeesh/yogeesh/datasets/car/data/train_solution_bounding_boxes.csv"
    data_path = "/home/yogeesh/yogeesh/datasets/car/data/training_images"
    train_obj = ResnetTrainer(data_path , true_csv_path , 0.001 , 32 , 2)
    train_obj.train()
