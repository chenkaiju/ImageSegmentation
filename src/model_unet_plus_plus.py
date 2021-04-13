import tensorflow as tf
import os
import numpy as np

#from tensorflow_examples.models.pix2pix import pix2pix

def sconv_block(x, filters, kernel=(3, 3), strides=1, activation='swish', name=None):
    x = tf.keras.layers.Activation(activation, name=name + '_sep_activation')(x)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size=kernel, strides=strides, padding='same', name=name + '_sep_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'_sep_bn')(x)
    return x

def U(x, ref, name):
  num_filters = int(ref.shape.as_list()[-1])
  x = tf.keras.layers.UpSampling2D((2, 2), interpolation='nearest', name=name + '_upsampling')(x)
  x = sconv_block(x, num_filters, name=name+'_Up')
  return x

def C(x, name):
  #concat = tf.keras.layers.Concatenate()
  return tf.keras.layers.concatenate(x, axis=-1, name=name)


class UNetPlusPlusModel():
  def __init__(self, output_channels = 3, checkpoint_dir='checkpoint', summary_folder = 'logs'):
    
    self.model = None
    self.training_dataset = None
    self.validation_dataset = None

    self.output_channels = output_channels

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.checkpoint_dir = checkpoint_dir

    self.summary_folder = summary_folder

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.summary_folder, 
                                                       profile_batch=0, 
                                                       write_images=True)

    savecheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_dir, 'model.h5'), 
                                                     monitor='loss', 
                                                     save_best_only=True, mode='min')

    self.callback = [tensorboard_callback, savecheckpoint_callback]

  def ResumeModel(self, model_path=None):
    self.model = tf.keras.models.load_model(model_path, compile=True)

    return self.model is not None    
  
  def customized_callback(self, callback_list):
        self.callback.extend(callback_list)

  def BuildModel(self, learning_rate=1e-4): # Model definition

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
      
    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    down_stack.trainable = False

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    X00, X10, X20, X30, X40 = down_stack(inputs)
      
    X01 = C([U(X10, X00, 'X10_UP'), X00], 'X01')
    X11 = C([U(X20, X10, 'X20_UP'), X10], 'X11')
    X21 = C([U(X30, X20, 'X30_UP'), X20], 'X21')
    X31 = C([U(X40, X30, 'X40_UP'), X30], 'X31')

    X02 = C([U(X11, X01, 'X11_UP'), X00, X01], 'X02')
    X12 = C([U(X21, X11, 'X21_UP'), X10, X11], 'X12')  
    X22 = C([U(X31, X21, 'X31_UP'), X20, X21], 'X22')

    X03 = C([U(X12, X02, 'X12_UP'), X00, X01, X02], 'X03')
    X13 = C([U(X22, X12, 'X22_UP'), X10, X11, X12], 'X13')

    X04 = C([U(X13, X03, 'X13_UP'), X00, X01, X02, X03], 'X04')

    last = tf.keras.layers.Conv2DTranspose(self.output_channels, 3, strides=2, padding='same')  #64x64 -> 128x128
    output = last(X04)

    self.model = tf.keras.Model(inputs=inputs, outputs=output)
    self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    self.model.summary()
    tf.keras.utils.plot_model(self.model, to_file='unet_simplified.png', show_shapes=True)

    return self.model

  def train(self, epochs = 1, steps_per_epoch = 1, validation_steps = 1, train_data = None, validation_data = None):

    #if (validation_data is not None):
    #  self.callback.extend([ModelCustomedCallback(self.summary_writer, validation_data)])

    history = self.model.fit(x=train_data, 
                          epochs=epochs,
                          steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps,
                          validation_data=validation_data,
                          callbacks=self.callback)

    return history
  
  def predict(self, dataset=None):
    return self.model.predict(dataset)






