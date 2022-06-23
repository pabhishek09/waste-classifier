import { readFileSync } from 'fs'
import { readdir } from 'fs/promises'
import { dirname, join } from 'path'
import predict from './predict.mjs'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const sampleDatasetPath = join(__dirname, 'sample-dataset')

const sampleImages = await readdir(sampleDatasetPath)

sampleImages
  .map(path => join(sampleDatasetPath, '/', path))
  .forEach(async(imgPath) => {
    const bitmap = readFileSync(imgPath)
    const prediction = await predict(new Uint8Array(bitmap))
    console.log(`Prediction for ${imgPath} is : ${JSON.stringify(prediction)}`)
  })
