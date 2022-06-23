# Waste classifier

Builds a  convolutional neural network  to identify the waste type using the TensorFlow framework

Uses the following the [Dataset](https://data.mendeley.com/datasets/n3gtgm9jxj/2) to train the model

## Get started

1. Load data - Run `npm run fetch-datset`
2. Train the model - Run `npm run train-model`, run `npm run train-model:dev` to train on fewer datasets and generate a minimalist model
3. Test predictions - Run `npm run test-prediction` to test the predictions locally on sample dataset
4. Run npm start to start the node app which exposes a `/predict` endpoint 

