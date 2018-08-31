# ResNet in TensorFlow
Please proceed according to which dataset you would like to train/evaluate on:


## CIFAR-10

### Setup

Simply run : 
bash train_cifar10.sh 

or follow these instructions: 

You simply need to have the latest version of TensorFlow installed.
First make sure you've [added the models folder to your Python path](/official/#running-the-models); otherwise you may encounter an error like `ImportError: No module named official.resnet`.

Then download and extract the CIFAR-10 data from Alex's website, specifying the location with the `--data_dir` flag. Run the following:

```
python cifar10_download_and_extract.py
```

Then to train the model, run the following:

```
python cifar10_main.py
```

Use `--data_dir` to specify the location of the CIFAR-10 data used in the previous step. There are more flag options as described in `cifar10_main.py`.

