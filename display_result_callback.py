import tensorflow as tf

# Customized callback
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = tf.cast(pred_mask, tf.float32)
  cond1 = tf.equal(pred_mask, 0.0)
  cond2 = tf.equal(pred_mask, 1.0)
  cond3 = tf.equal(pred_mask, 2.0)

  bcast = tf.ones(tf.shape(pred_mask), dtype=pred_mask.dtype)
  gcast = tf.ones(tf.shape(pred_mask), dtype=pred_mask.dtype)
  rcast = tf.ones(tf.shape(pred_mask), dtype=pred_mask.dtype)

  class1 = tf.where(cond1, bcast, tf.zeros_like(pred_mask))
  class2 = tf.where(cond2, gcast, tf.zeros_like(pred_mask))
  class3 = tf.where(cond3, rcast, tf.zeros_like(pred_mask))

  trimap = tf.stack([class1, class2, class3], axis=3)

  return trimap

def create_true_mask(true_mask):

  cond1 = tf.equal(true_mask, 0.0) # background
  cond2 = tf.equal(true_mask, 1.0) # foreground
  cond3 = tf.equal(true_mask, 2.0) # unknown

  bcast = tf.ones(tf.shape(true_mask), dtype=true_mask.dtype)
  gcast = tf.ones(tf.shape(true_mask), dtype=true_mask.dtype)
  rcast = tf.ones(tf.shape(true_mask), dtype=true_mask.dtype)

  class1 = tf.where(cond1, rcast, tf.zeros_like(true_mask))
  class2 = tf.where(cond2, gcast, tf.zeros_like(true_mask))
  class3 = tf.where(cond3, bcast, tf.zeros_like(true_mask))

  trimap = tf.concat([class1, class2, class3], axis=3)

  return trimap

class display_result_callback(tf.keras.callbacks.Callback):
  def __init__(self, summary_writer=None, validation_data=None):
    self.writer = summary_writer
    self.validation_dataset = validation_data

  def on_epoch_end(self, epoch, logs=None):
    
    if self.validation_dataset:
      
      num = 1
      data = self.validation_dataset.take(num)

      tbimage1 = []
      for image, true_mask in data:

        pred_mask = self.model.predict(image)
        tbimage = create_mask(pred_mask)
        tbimage2 = create_true_mask(true_mask)
        tbimages1 = tf.concat([image, tbimage, tbimage2], axis=2)

      with self.writer.as_default():
        tf.summary.image('test image', tbimages1, max_outputs=8, step=epoch)
    
