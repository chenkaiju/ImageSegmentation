import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import os

class UNetModel():
    def __init__(self, output_channels=3, checkpoint_dir='checkpoint', summary_folder = 'logs'):
        
        self._model = None
        self._training_dataset = None
        self._validation_dataset = None
        self._output_channels = output_channels

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self._checkpoint_dir = checkpoint_dir
        self._summary_folder = summary_folder

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self._summary_folder, 
                                                       profile_batch=0, 
                                                       write_images=True)

        savecheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(self._checkpoint_dir, 'model.h5'), 
                                                     monitor='loss', 
                                                     save_best_only=True, mode='min')

        self._callback = [tensorboard_callback, savecheckpoint_callback]   

        self._early_stop_callback = None

    def ResumeModel(self, model_path=None):

        self._model = tf.keras.models.load_model(model_path, compile=True)
        return self._model is not None           

    def customized_callback(self, callback_list):

        self._callback.extend(callback_list)
    
    def enable_early_stopping(self):

        self._early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        self._callback.extend([self._early_stop_callback])

    def BuildModel(self, learning_rate=10e-4): 

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

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]
        
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            self._output_channels, 3, strides=2,
            padding='same')  #64x64 -> 128x128

        x = last(x)
        self._model = tf.keras.Model(inputs=inputs, outputs=x)

        self._model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self._model.summary()
        tf.keras.utils.plot_model(self._model, to_file='unet.png', show_shapes=True)

        return self._model

    def train(self, epochs = 1, steps_per_epoch = 1, validation_steps = 1, train_data = None, validation_data = None, 
              enable_early_stopping=False):

        if enable_early_stopping is True:
          self.enable_early_stopping()

        history = self._model.fit(x=train_data, 
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_data=validation_data,
                              callbacks=self._callback)

        return history
  
    def predict(self, dataset=None):

        return self._model.predict(dataset)