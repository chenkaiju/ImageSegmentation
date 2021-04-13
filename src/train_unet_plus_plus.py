#%%
import tensorflow as tf
#from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_datasets as tfds
#from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
from model_unet_plus_plus import UNetPlusPlusModel
from display_result_callback import display_result_callback
from display_util import create_mask, create_true_mask
# utility functions
# def create_mask(pred_mask):
#   pred_mask = tf.argmax(pred_mask, axis=-1)
#   pred_mask = pred_mask[..., tf.newaxis]
#   return pred_mask

def display(display_list, save = False, path = None):

  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    image = tf.keras.preprocessing.image.array_to_img(display_list[i])
    plt.imshow(image)
    plt.axis('off')
  
  plt.show()
  if save==True:
    plt.savefig(path)

def show_predictions(model=None, dataset=None, num=1, save=False, save_folder=None):
  if save_folder is None:
      save = False

  if dataset:
 
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      color_mask = create_true_mask(mask)
      color_pred_mask = create_mask(pred_mask)
      batch_size = image.shape[0]
      for i in range(batch_size):
        path = os.path.join(save_folder, ''.join([str(i), '.jpg']))
        
        display([image[i], color_mask[i], color_pred_mask[i]], save=save, path=path)




#%% Process training data...

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1   # change label to 0-index
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

@tf.function
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 8
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)


train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

# for image, mask in train.take(1):
#   sample_image, sample_mask = image, mask
# display([sample_image, sample_mask])

# %%
# Define Model

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
OUTPUT_CHANNELS = 3
RESUME = False
TRAIN = True

checkpoint_dir = 'check_point'
summary_folder = 'logs'
model = UNetPlusPlusModel(output_channels=OUTPUT_CHANNELS, checkpoint_dir=checkpoint_dir, summary_folder=summary_folder)
if RESUME :
    success = model.ResumeModel(model_path='check_point/model.h5')
    if not success:
        model.BuildModel()
else:
    model.BuildModel()

summary_writer = tf.summary.create_file_writer(os.path.join(summary_folder,'test'))
model.customized_callback([display_result_callback(summary_writer=summary_writer, validation_data = test_dataset)])

if TRAIN:
    model_history = model.train(train_data=train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset)

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']



#%% Make prediction
prediction_folder = os.path.abspath(os.path.join('./', 'unet_plus_plus'))
if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

show_predictions(model=model, dataset=test_dataset, num=1, save=True, save_folder=prediction_folder)
# %%
