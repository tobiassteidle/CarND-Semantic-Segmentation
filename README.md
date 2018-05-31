# Semantic Segmentation
Self-Driving Car Engineer Nanodegree Program

##### Image Preview
![Video](images/semantic_segmentation.gif)

##### Result Images (Samples)
![Sample1](images/runs/umm_000006.png)
![Sample2](images/runs/umm_000015.png)
![Sample2](images/runs/umm_000034.png)
![Sample2](images/runs/umm_000050.png)

##### Tensorflow Graph
![Tensorflow](images/tensorflow_graph.png)

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
 
##### Install as Conda environment
```
conda env create -f environment.yml
```
```
conda activate carnd-semantic-segmentation
```

##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```


