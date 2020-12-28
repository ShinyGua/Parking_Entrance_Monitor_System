#! /bin/bash
sudo apt-get update
sudo apt-get install python3.8
sudo apt install python3-dev python3-pip python3-venv
sudo apt install git-all
sudo apt install python3-opencv
pip3 install --upgrade tensorflow
pip3 install -q tensorflow-model-optimization
git clone https://gitlab.anu.edu.au/u1063268/parking-systems.gitk