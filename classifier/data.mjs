import * as tf from '@tensorflow/tfjs-node-gpu'
import { fileURLToPath } from 'url'
import { existsSync, readFileSync } from 'fs'
import { readdir } from 'fs/promises'
import { dirname, join } from 'path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

async function get (develop) {
  const organicDatasetPath = join(__dirname, 'waste-dataset/organic')
  const recyclableDatasetPath = join(__dirname, 'waste-dataset/recyclable')

  if (!existsSync(organicDatasetPath) || !existsSync(recyclableDatasetPath)) throw new Error('Run the fetch-dataset script')

  let organicDataset
  try {
    organicDataset = await readdir(organicDatasetPath)
    organicDataset = organicDataset.map(file => organicDatasetPath.concat('/', file))
  } catch (err) {
    throw new Error(`Error in reading organic dataset ${err}`)
  }

  let recyclableDataset
  try {
    recyclableDataset = await readdir(recyclableDatasetPath)
    recyclableDataset = recyclableDataset.map(file => recyclableDatasetPath.concat('/', file))
  } catch (err) {
    throw new Error(`Error in reading recycable dataset ${err}`)
  }

  const trainingFraction = 0.85

  const organicTrainingIndex = Math.round(trainingFraction * organicDataset.length)
  const recyclableTrainingIndex = Math.round(trainingFraction * recyclableDataset.length)

  let trainFeatures = develop
    ? organicDataset.slice(0, 10)
    : organicDataset.slice(0, organicTrainingIndex)
  let trainTarget = develop ? new Array(10).fill(0) : new Array(organicTrainingIndex).fill(0)

  let testFeatures = develop ? organicDataset.slice(10, 12) : organicDataset.slice(organicTrainingIndex)
  let testTarget = develop
    ? new Array(2).fill(0)
    : new Array(organicDataset.length - organicTrainingIndex).fill(0)

  trainFeatures = trainFeatures.concat(
    develop
      ? recyclableDataset.slice(0, 10)
      : recyclableDataset.slice(0, recyclableTrainingIndex)
  )
  trainTarget = trainTarget.concat(develop ? new Array(10).fill(1) : new Array(recyclableTrainingIndex).fill(1))
  testFeatures = testFeatures.concat(develop
    ? recyclableDataset.slice(10, 12)
    : recyclableDataset.slice(recyclableTrainingIndex))
  testTarget = testTarget.concat(develop ? new Array(2).fill(1) : new Array(recyclableDataset.length - recyclableTrainingIndex).fill(1))

  tf.util.shuffleCombo(trainFeatures, trainTarget)

  console.log({
    trainFeatures,
    trainTarget
  })

  trainFeatures = trainFeatures
    .map(path => tf.tidy(() => tf.node.decodeImage(readFileSync(path))))
    .map(imgTensor => tf.image.resizeNearestNeighbor(imgTensor, [224, 224], true))
  testFeatures = testFeatures
    .map(path => tf.tidy(() => tf.node.decodeImage(readFileSync(path))))
    .map(imgTensor => tf.image.resizeNearestNeighbor(imgTensor, [224, 224], true))

  return {
    train: {
      images: tf.stack(trainFeatures),
      labels: tf.oneHot(trainTarget, 2).toFloat()
    },
    test: {
      images: tf.stack(testFeatures),
      labels: tf.oneHot(testTarget, 2).toFloat()
    }
  }
}

export default get
