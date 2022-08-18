# Introduction
Most of the code is inherited from [DBNet](https://github.com/MhLiao/DB). More details can be found 
in [DBNet](https://github.com/MhLiao/DB).


## Installation
### Requirements:
- Python3
- PyTorch == 1.8
- GCC >= 4.9 (This is important for PyTorch)
- CUDA >= 9.0 (10.1 is recommended)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name DB -y
  conda activate DB

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install -r requirement.txt

  # install PyTorch with cuda-10.1
  conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

  # clone repo
  git clone https://github.com/MhLiao/DB.git
  cd DB/

  # build deformable convolution opertor
  # make sure your cuda path of $CUDA_HOME is the same version as your cuda in PyTorch
  # make sure GCC >= 4.9
  # you need to delete the build directory before you re-build it.
  echo $CUDA_HOME
  cd assets/ops/dcn/
  python setup.py build_ext --inplace

```

## Datasets
The root of the dataset directory can be ```DB/datasets/```.

Download the converted ground-truth and data list [Baidu Drive](https://pan.baidu.com/s/1BPYxcZnLXN87rQKmz9PFYA) (download code: mz0a), [Google Drive](https://drive.google.com/open?id=12ozVTiBIqK8rUFWLUrlquNfoQxL2kAl7). The images of each dataset can be obtained from their official website.

## Testing
### Prepar dataset
An example of the path of test images: 
```
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt
```
The data root directory and the data list file can be defined in ```base_totaltext.yaml```

### Config file
**The YAML files with the name of ```base*.yaml``` should not be used as the training or testing config file directly.**


## Training
Check the paths of data_dir and data_list in the base_*.yaml file. For better performance, you can first per-train the model with SynthText and then fine-tune it with the specific real-world dataset.

```CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py path-to-yaml-file --num_gpus 4```

You can also try distributed training (**Note that the distributed mode is not fully tested. I am not sure whether it can achieves the same performance as non-distributed training.**)

```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py path-to-yaml-file --num_gpus 4```

## Improvements
Note that the current implementation is written by pure Python code except for the deformable convolution operator. Thus, the code can be further optimized by some optimization skills, such as [TensorRT](https://github.com/NVIDIA/TensorRT) for the model forward and efficient C++ code for the [post-processing function](https://github.com/MhLiao/DB/blob/d0d855df1c66b002297885a089a18d50a265fa30/structure/representers/seg_detector_representer.py#L26).

Another option to increase speed is to run the model forward and the post-processing algorithm in parallel through a producer-consumer strategy.

Contributions or pull requests are welcome.
