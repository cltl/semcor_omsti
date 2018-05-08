#!/usr/bin/env bash


rm -rf resources

mkdir resources
cd resources
wget http://lcl.uniroma1.it/wsdeval/data/WSD_Unified_Evaluation_Datasets.zip
unzip WSD_Unified_Evaluation_Datasets.zip

wget http://lcl.uniroma1.it/wsdeval/data/WSD_Training_Corpora.zip
unzip WSD_Training_Corpora.zip