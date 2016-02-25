#!/bin/bash

# Check if lua is properly installed
luarockspath=`which luarocks`
uname=`whoami`
torchbinpath=/home/scratch/$uname/torch/bin

if [ "$luarockspath" == '' ]
then
	if [ ! -f "$torchbinpath/luarocks" ]
	then
		echo "Cannot find luarocks!"
		echo "Run the install script from practical1 before running this script"
		exit
	else
		export PATH=$torchbinpath:$PATH
	fi
fi

# Luarocks seems to be installed fine
# Check if svm already installed
SVMINSTALL=`luarocks list | grep "^svm"`
if [ "$SVMINSTALL" == "svm" ]
then
	echo "You already have torch-svm installed, proceeding to download data"
else
	echo "Installing torch-svm"
	luarocks install svm
fi

# Set up data for practical 4
mkdir -p practical4-data/raw-data
echo "Downloading Training Data"
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2 
echo "Downloading Training Data"
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzcat mnist.scale.bz2 > practical4-data/train_data
bzcat mnist.scale.t.bz2 > practical4-data/test_data
mv mnist.scale.bz2 mnist.scale.t.bz2 practical4-data/raw-data

echo "The training and test data has been downloaded in the folder practical4-data"
if [ "$luarockspath" == '' ]
then
	echo "You may want to add the export PATH=$torchbinpath:\$PATH to your .bashrc"
	echo "Then run source ~/.bashrc"
fi
