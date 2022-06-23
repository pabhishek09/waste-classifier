import * as tf from '@tensorflow/tfjs-node-gpu'

function create () {
  const model = tf.sequential()

  const IMAGE_WIDTH = 224
  const IMAGE_HEIGHT = 224
  const IMAGE_CHANNELS = 3

  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    filters: 64,
    kernelSize: 3,
    padding: 'same',
    strides: 1,
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    padding: 'same',
    strides: 1,
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  model.add(tf.layers.conv2d({
    filters: 256,
    kernelSize: 3,
    padding: 'same',
    strides: 1,
    activation: 'relu'
  }))
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

  model.add(tf.layers.flatten())
  model.add(tf.layers.dense({ units: 256, activation: 'tanh' }))
  // model.add(tf.layers.dropout({ rate: 0.5 }))
  model.add(tf.layers.dense({ units: 64, activation: 'tanh' }))
  // model.add(tf.layers.dropout({ rate: 0.5 }))
  model.add(tf.layers.dense({ units: 2, activation: 'softmax' }))

  model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
  })

  return model
}

export default create
