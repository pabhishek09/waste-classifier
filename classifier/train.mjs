import { fileURLToPath } from 'url'
import { dirname } from 'path'
import get from './data.mjs'
import create from './model.mjs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const develop = process.env.NODE_ENV === 'dev'

async function save (epochs, batchSize, modelSavePath) {
  const { train, test } = await get(develop)
  const { images: trainImages, labels: trainLabels } = train

  const model = create()
  model.summary()

  console.log({
    trainImages,
    trainLabels
  })

  const validationSplit = 0.15

  await model.fit(trainImages, trainLabels, {
    epochs,
    batchSize,
    validationSplit,
    shuffle: true
  })

  const { images: testImages, labels: testLabels } = test

  const evalOutput = model.evaluate(testImages, testLabels)

  console.log(`Evaluation result: Loss = ${evalOutput[0].dataSync()[0].toFixed(3)} Accuracy = ${evalOutput[1].dataSync()[0].toFixed(3)}`)

  if (modelSavePath != null) {
    await model.save(`file://${modelSavePath}`)
    console.log(`Saved model to path: ${modelSavePath}`)
  }
}

const EPOCHS = 20
const BATCH_SIZE = 256
const SAVE_PATH = `${__dirname}/model`

save(EPOCHS, BATCH_SIZE, SAVE_PATH)
