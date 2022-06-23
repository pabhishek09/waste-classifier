import express from 'express'
import predict from '../classifier/predict.mjs'

const router = express.Router()

router.post('/', async function (req, res, next) {
  console.log(req.files.upload)

  const prediction = await predict(new Uint8Array(req.files.upload.data))
  res.send({prediction: prediction || 'unsupported file type' })
})

export default router
