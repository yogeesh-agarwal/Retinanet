import tensorflow as tf
from layers import ConvBlock , ResnetBlock

class Resnet50(tf.keras.Model):
    def __init__(self , input_size):
        super(Resnet50, self).__init__(name  = "Resnet50_backbone")
        print("Initializing Resnet50 as backbone")

        self.conv1 = ConvBlock(True , True , 7 , 2 , True , 64)
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size = (3,3) , strides = (2,2), padding = "SAME")

        self.res_block2 = []
        self.res_block2.append(ResnetBlock(True , [1 , 1 , 3, 1] , [256, 64, 64, 256] , [False , False , True , False] , [1, 1, 1, 1]))
        for i in range(2):
            self.res_block2.append(ResnetBlock(False , [1, 3, 1] , [64, 64, 256] , [False , True, False] , [1, 1, 1]))

        self.res_block3 = []
        self.res_block3.append(ResnetBlock(True , [1 , 1, 3, 1] , [512 , 128 , 128 , 512] , [False , False , True , False] , [2 , 2 , 1 , 1]))
        for i in range(3):
            self.res_block3.append(ResnetBlock(False , [1 , 3 , 1] , [128 , 128 , 512] , [False , True , False] , [1, 1, 1]))

        self.res_block4 = []
        self.res_block4.append(ResnetBlock(True , [1, 1, 3, 1] , [1024 , 256 , 256, 1024] , [False , False, True, False] , [2 , 2 , 1, 1]))
        for i in range(5):
            self.res_block4.append(ResnetBlock(False , [1 , 3 , 1] , [256 , 256 , 1024] , [False , True , False] , [1, 1, 1]))

        self.res_block5 = []
        self.res_block5.append(ResnetBlock(True , [1, 1, 3, 1] , [2048, 512 , 512, 2048], [False, False, True, False], [2, 2, 1, 1]))
        for i in range(2):
            self.res_block5.append(ResnetBlock(False , [1 , 3 , 1] , [512 , 512 , 2048] , [False , True , False] , [1, 1, 1]))

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

        return [res_block2 , res_block3 , res_block4 , res_block5]

    def build_graph(self):
        x = tf.keras.Input(shape = [224 , 224 , 3])
        return tf.keras.Model(inputs = [x] , outputs = self.call(x))

def test_model():
    resnet50 = Resnet50(224)
    resnet50.build(input_shape = [1,224,224,3])
    resnet50.build_graph().summary()
    # tf.keras.utils.plot_model(resnet50.build_graph(), to_file='./arch_png/resnet50.png', show_shapes=True, show_dtype=False,
    #                           show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96,
    #                           layer_range=None)

test_model()
