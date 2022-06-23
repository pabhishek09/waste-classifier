#!/bin/sh

cd ./classifier                           
curl -sS https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/n3gtgm9jxj-2.zip > dataset.zip
unzip dataset.zip 
mkdir -p ./waste-dataset/organic  && mkdir ./waste-dataset/recyclable 
mv ./n3gtgm9jxj-2/waste_dataset/organic ./waste-dataset && mv ./n3gtgm9jxj-2/waste_dataset/recyclable ./waste-dataset
rm -rf ./n3gtgm9jxj-2
rm ./dataset.zip
