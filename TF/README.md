place  this folder into *tf_models/official* next to other folders such as resnet, mnist, etc. then follow the instructions below: 
---
## Running the models

The *Official Models* are made available as a Python module. To run the models and associated scripts, add the top-level ***/models*** folder to the Python path with the command: `export PYTHONPATH="$PYTHONPATH:/path/to/models"`

To install dependencies pass `-r official/requirements.txt` to pip. (i.e. `pip3 install --user -r official/requirements.txt`)

To make Official Models easier to use, we are planning to create a pip installable Official Models package. This is being tracked in [#917](https://github.com/tensorflow/models/issues/917).
