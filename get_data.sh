#!/usr/bin/env bash

wget https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip
unzip hyper-kvasir-labeled-images.zip
python prepare_endo_data.py
#rm hyper-kvasir-labeled-images.zip
#rm -r data/labeled-images/

