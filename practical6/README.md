# Practical 6
Machine Learning, Hilary Term 2016

In this practical, we will experiment with PCA

## Setup
Setup will be the same as in practical 1. Please refer to the [practical 1 repository](https://github.com/oxford-cs-ml-2016/practical1), and run the script as instructed.

We will use the Torch package `gnuplot` for plotting; we will also use the package `image` to convert PCs and reconstructions into images. If you don't have the image package installed run 

```
luarocks install image
```

## Practical
See the writeup PDF for instructions. Be sure to clone the repository instead of downloading file-by-file so you don't miss any:

```
bash 
git clone https://github.com/oxford-cs-ml-2016/practical6
cd practical6
```

The file `p6_process_data.lua` will normalise the data, perform svd, and store these on disk. The file `practical6.lua` reads the data from the created files.


# See course page for practicals
<https://www.cs.ox.ac.uk/people/varun.kanade/ML2016/>
