from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import datetime

from dataset_builder import DatasetBuilder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


base_model = tf.keras.applications.ResNet101V2(input_shape=(224, 224, 3),
                           include_top=False,
                           weights='imagenet')
    
base_model.trainable = True
fine_tune_at = 300
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False


def build_model(params, num_classes):
  image_shape = params['image_shape']

  image_input = keras.Input(shape=image_shape)
  x = base_model(image_input)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(num_classes, activation='softmax')(x)
  model = keras.Model(inputs=image_input, outputs=x)

  model.compile(optimizer=keras.optimizers.Adam(0.0001),
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model


def train_and_evaluate(params):
  logging.info('Running train and evaluate fn.')

  batch_size = params['batch_size']
  num_epochs = params['num_epochs']
  output_dir = params['output_dir']
  timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  savedmodel_dir = os.path.join(output_dir, 'savedmodel')
  model_export_path = os.path.join(savedmodel_dir, timestamp)
  
  if tf.io.gfile.exists(output_dir):
    tf.io.gfile.rmtree(output_dir)

  builder = DatasetBuilder(params)

  train_ds, val_ds = builder.create_dataset()

  class_names = builder.class_names
  num_classes = len(class_names)

  train_steps = builder.train_steps
  val_steps = builder.val_steps

  model = build_model(params, num_classes)
  
  history = model.fit(
      train_ds,
      batch_size=batch_size,
      epochs=num_epochs,
      validation_data=val_ds,
      steps_per_epoch=train_steps,
      validation_steps=val_steps)
  
  tf.saved_model.save(model, model_export_path)
                      
  return history