import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self , relu , bn , kernel_size , stride , padding , num_filters , index , **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.bn = bn
        self.relu = relu
        self.index = index
        self.stride = stride
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = "SAME" if padding else "VALID"

    def build(self, input_shape):
        #TODO : add pretrain logic to load prtrain conv and bn weights.
        if not self.kernel_size:
            raise Exception("kernel size is required to perform convolution")

        self.kernel = self.add_weight(shape =  [self.kernel_size , self.kernel_size , input_shape[-1] , self.num_filters],
                                      initializer = "random_normal" ,
                                      dtype = tf.float32 ,
                                      trainable = True ,
                                      name = "weights_{}".format(self.index))

        if self.bn:
            self.batch_norm = tf.keras.layers.BatchNormalization()

        else:
            self.bias = self.add_weight(shape = [self.num_filters],
                                        initializer = "zeros" ,
                                        dtype = tf.float32 ,
                                        trainable = True ,
                                        bias = "bias_{}".format(self.index))
        super(ConvBlock , self).build(input_shape = input_shape)

    def call(self, inputs):
        strides = [1 , self.stride , self.stride , 1]
        x = tf.nn.conv2d(inputs , self.kernel , strides , self.padding)
        if self.bn:
            x = self.batch_norm(x , training = True)

        else:
            x = tf.nn.bias_add(x , self.bias)

        if self.relu:
            x = tf.nn.relu(x)
        return x

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self , residual , kernel_sizes , num_filters , paddings , strides , index , **kwargs):
        super(ResnetBlock , self).__init__(**kwargs)
        num_layers = 4 if residual else 3
        if len(kernel_sizes) != len(num_filters) != len(paddings) != len(strides) != num_layers:
            raise Exception("{} layer informantion is required. got only {}".format(num_layers , len(kernel_size)))

        # index is global conv_layer index
        #layer_index is local index to unpack conv_layer info.
        self.residual = residual
        layer_index = 0
        if self.residual:
            self.left_conv = ConvBlock(False , True , kernel_sizes[layer_index] , strides[layer_index] , paddings[layer_index] , num_filters[layer_index] , index)
            index += 1
            layer_index += 1
        self.right_conv_1 = ConvBlock(True , True , kernel_sizes[layer_index] , strides[layer_index] , paddings[layer_index] , num_filters[layer_index] , index)
        index += 1
        layer_index += 1
        self.right_conv_2 = ConvBlock(True , True , kernel_sizes[layer_index] , strides[layer_index] , paddings[layer_index] , num_filters[layer_index] , index)
        index += 1
        layer_index += 1
        self.right_conv_3 = ConvBlock(False , True , kernel_sizes[layer_index] , strides[layer_index] , paddings[layer_index] , num_filters[layer_index] , index)
        index += 1
        layer_index += 1

    def call(self , inputs):
        if self.residual:
            left_branch  = self.left_conv(inputs)
        else:
            left_branch = inputs
        right_branch = self.right_conv_1(inputs)
        right_branch = self.right_conv_2(right_branch)
        right_branch = self.right_conv_3(right_branch)

        output = tf.nn.relu(tf.add(left_branch , right_branch))
        return output
