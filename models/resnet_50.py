import tensorflow as tf
from Retinanet.models.layers import ConvBlock , ResnetBlock

class Resnet50(tf.keras.Model):
    def __init__(self , input_size , include_top = False , num_classes = None):
        super(Resnet50, self).__init__(name  = "Resnet50_backbone")
        print("Initializing Resnet50 as backbone")
        self.index = 0
        self.include_top = include_top
        if include_top:
            self.num_classes = num_classes

        #define loss function and metrics
        self.resnet_loss = tf.keras.losses.CategoricalCrossentropy()
        self.train_loss_tracker = tf.keras.metrics.Mean(name = "train_loss")
        self.train_acc = tf.keras.metrics.Accuracy(name = "train_acc")
        self.val_acc = tf.keras.metrics.Accuracy(name = "val_acc")

        #define resnet model
        self.conv1 = ConvBlock(True , True , 7 , 2 , True , 64, self.index)
        self.index += 1
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size = (3,3) , strides = (2,2), padding = "SAME")

        self.res_block2 = []
        self.res_block2.append(ResnetBlock(True , [1 , 1 , 3, 1] , [256, 64, 64, 256] , [False , False , True , False] , [1, 1, 1, 1] , self.index))
        self.index += 4
        for i in range(2):
            self.res_block2.append(ResnetBlock(False , [1, 3, 1] , [64, 64, 256] , [False , True, False] , [1, 1, 1] , self.index))
            self.index += 3

        self.res_block3 = []
        self.res_block3.append(ResnetBlock(True , [1 , 1, 3, 1] , [512 , 128 , 128 , 512] , [False , False , True , False] , [2 , 2 , 1 , 1] , self.index))
        self.index += 4
        for i in range(3):
            self.res_block3.append(ResnetBlock(False , [1 , 3 , 1] , [128 , 128 , 512] , [False , True , False] , [1, 1, 1] , self.index))
            self.index += 3

        self.res_block4 = []
        self.res_block4.append(ResnetBlock(True , [1, 1, 3, 1] , [1024 , 256 , 256, 1024] , [False , False, True, False] , [2 , 2 , 1, 1] , self.index))
        self.index += 4
        for i in range(5):
            self.res_block4.append(ResnetBlock(False , [1 , 3 , 1] , [256 , 256 , 1024] , [False , True , False] , [1, 1, 1] , self.index))
            self.index += 3

        self.res_block5 = []
        self.res_block5.append(ResnetBlock(True , [1, 1, 3, 1] , [2048, 512 , 512, 2048], [False, False, True, False], [2, 2, 1, 1] , self.index))
        self.index += 4
        for i in range(2):
            self.res_block5.append(ResnetBlock(False , [1 , 3 , 1] , [512 , 512 , 2048] , [False , True , False] , [1, 1, 1] , self.index))
            self.index += 3

        if self.include_top:
            self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size = (7,7) , strides = (1,1))
            self.fc = tf.keras.layers.Dense(units = self.num_classes , activation = "softmax")

    def call(self , inputs):
        conv1 = self.conv1(inputs)
        max_pool = self.max_pool(conv1)

        res_block2 = self.res_block2[0](max_pool)
        for i in range(1 , 3):
            res_block2 = self.res_block2[i](res_block2)

        res_block3 = self.res_block3[0](res_block2)
        for i in range(1 , 4):
            res_block3 = self.res_block3[i](res_block3)

        res_block4 = self.res_block4[0](res_block3)
        for i in range(1 , 6):
            res_block4 = self.res_block4[i](res_block4)

        res_block5 = self.res_block5[0](res_block4)
        for i in range(1 , 3):
            res_block5 = self.res_block5[i](res_block5)

        if self.include_top:
            avg_pool = self.avg_pool(res_block5)
            output = self.fc(avg_pool)
            output = tf.reshape(output , shape = [tf.shape(output)[0] , self.num_classes])
            return output

        return [res_block2 , res_block3 , res_block4 , res_block5]

    def train_step(self , data):
        images , labels = data
        with tf.GradientTape() as tape:
            preds = self(images , training = True)
            loss = self.resnet_loss(labels , preds)

        # apply gradients on trainable vars
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss , trainable_vars)
        self.optimizer.apply_gradients(zip(gradients , trainable_vars))

        # compute metrics
        self.train_loss_tracker.update_state(loss)
        self.train_acc.update_state(labels , preds)
        return {"train_loss" : self.train_loss_tracker.result() ,
                "train_accuracy" : self.train_acc.result()}

    def test_step(self , data):
        images , labels = data
        preds = self(images , training = False)
        loss = self.resnet_loss(labels , preds)
        self.val_acc.update_state(labels , preds)
        return {"val_accuracy" : self.val_acc.result()}

    @property
    def metrics(self):
        return [self.train_loss_tracker , self.train_acc , self.val_acc]

    def build_graph(self):
        x = tf.keras.Input(shape = [224 , 224 , 3])
        return tf.keras.Model(inputs = [x] , outputs = self.call(x))

def test_model():
    resnet50 = Resnet50(224 , include_top = True , num_classes = 2)
    resnet50.build(input_shape = [1,224,224,3])
    resnet50.build_graph().summary()
    tf.keras.utils.plot_model(resnet50.build_graph(), to_file='./arch_png/resnet50.png', show_shapes=True, show_dtype=False,
                              show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96,
                              layer_range=None)
