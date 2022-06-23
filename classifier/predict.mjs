import * as tf from '@tensorflow/tfjs-node'
import path from 'path'
import { fileURLToPath } from 'url'
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const model = await tf.loadLayersModel(`file://${__dirname}/model/model.json`)

async function predict (imgUint8Array) {
  let prediction
  try {
    const tensor = tf.image.resizeNearestNeighbor(tf.node.decodeImage(imgUint8Array), [224, 224], true)
    prediction = await model.predict(tensor.expandDims())
  } catch(err) {
    console.log(err)
    return 'unsupported file type'
  }
  prediction.print()
  const predictionToArray = await prediction.array()
  const [organic, recyclable] = predictionToArray[0]
  return (organic > recyclable) ? 'organic' : 'recyclable'
}

export default predict
