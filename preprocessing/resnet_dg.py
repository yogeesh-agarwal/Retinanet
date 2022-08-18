import os
import csv
import cv2
import math
import numpy as np
import tensorflow as tf

class ResnetDataGenerator(tf.keras.utils.Sequence):
    def __init__(self , true_object_csv , data_path , batch_size, width = 224 , height = 224 , num_instances = 1000, num_classes = 2, shuffle = True , norm = True):
        self.norm = norm
        self.embeddings = []
        self.shuffle = shuffle
        self.inp_width = width
        self.inp_height = height
        self.data_path = data_path
        self.batch_size = batch_size
        self.true_objects_names = []
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.true_object_csv = true_object_csv
        self.read_csv()
        self.generate_embeddings()

    def read_csv(self):
        with open(self.true_object_csv , mode = "r") as f:
            csv_file = csv.reader(f)
            for line in csv_file:
                self.true_objects_names.append(line[0])

    def generate_embeddings(self):
        object_files = os.listdir(self.data_path)
        for object_name in object_files:
            if object_name in self.true_objects_names:
                curr_embedding = [object_name , [1,0]]

            else:
                curr_embedding = [object_name , [0,1]]
            self.embeddings.append(curr_embedding)

    def generate_data(self , starting_index , ending_index):
        curr_dataset = self.embeddings[starting_index : ending_index]
        images = []
        labels = []
        for data in curr_dataset:
            img_name = data[0]
            image_path = os.path.join(self.data_path , img_name)
            image = cv2.cvtColor(cv2.imread(image_path) , cv2.COLOR_BGR2RGB)
            image = cv2.resize(image , (self.inp_width , self.inp_height))
            if self.norm:
                image = image / 255.

            images.append(image)
            labels.append(data[1])

        return images , labels

    def load_data(self , index):
        starting_index = index * self.batch_size
        ending_index = starting_index + self.batch_size
        if ending_index > self.num_instances:
            ending_index = self.num_instances
            starting_index = ending_index - self.batch_size

        images , labels = self.generate_data(starting_index, ending_index)
        return images , labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.embeddings)

    def __len__(self):
        return math.ceil(self.num_instances / self.batch_size)

    def __getitem__(self , index):
        images , labels = self.load_data(index)
        return (np.array(images).reshape(self.batch_size, self.inp_height , self.inp_width , 3) ,
                np.array(labels).reshape(self.batch_size, self.num_classes))
