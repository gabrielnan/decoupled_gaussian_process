from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn import datasets

tf.logging.set_verbosity(tf.logging.INFO)


def nn_model_fn(features, labels, mode, params):
  layer_sizes = [10, 10]
  dropout_rates = [0.1, 0.1]
  in_dim = params['in_dim']  # todo: fix input size
  out_dim = params['out_dim']  # todo: fix output size
  learning_rate = 0.01

  # Input Layer
  input_layer = tf.reshape(features['x'], [-1, in_dim])

  # Layer 0
  dense0 = tf.layers.dense(inputs=input_layer,
                           units=layer_sizes[0],
                           activation=tf.nn.relu)
  dropout0 = tf.layers.dropout(inputs=dense0,
                               rate=dropout_rates[0],
                               training=mode == tf.estimator.ModeKeys.TRAIN)

  # Layer 1
  dense1 = tf.layers.dense(inputs=dropout0,
                           units=layer_sizes[1],
                           activation=tf.nn.relu)
  dropout1 = tf.layers.dropout(inputs=dense1,
                               rate=dropout_rates[1],
                               training=mode == tf.estimator.ModeKeys.TRAIN)

  # Output layer
  output = tf.layers.dense(inputs=dropout1, units=out_dim, name="output")

  predictions = {"output": output}

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate loss
  loss = tf.losses.mean_squared_error(labels, output)

  # Training Op
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Evaluate Op
  eval_metric_ops = {
    "MSE": tf.metrics.mean_squared_error(labels, output)}
  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  dataset = datasets.load_diabetes()
  n = dataset.data.shape[0]
  train_part = int(n * 0.7)  # Learn how to make .7 a hyperparameter

  x_train = dataset.data[:train_part]
  y_train = dataset.target[:train_part].reshape(-1, 1)
  x_test = dataset.data[train_part:]
  y_test = dataset.target[train_part:].reshape(-1, 1)

  in_dim = x_train.shape[1]
  out_dim = 1

  # Create Estimator
  nn = tf.estimator.Estimator(model_fn=nn_model_fn,
                              model_dir="/tmp/nn",
                              params={'in_dim': in_dim,
                                      'out_dim': out_dim})

  # Logging
  # tensors_to_log = {'output': 'output'}
  # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
  #                                           every_n_iter=50)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_train},
    y=y_train,
    batch_size=50,
    num_epochs=None,
    shuffle=True)

  nn.train(
    input_fn=train_input_fn,
    steps=1000,
    # hooks=[logging_hook]
  )

  # Final Evaluation
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': x_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
  eval_results = nn.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
