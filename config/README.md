#Configuration

##citypersons.yaml

This file contains all the key setup and hyperparameter settings for the training cycle the user wants to run next. It also includes the name 
of the directory that the model parameters and model output will be stored in, which I change before each training cycle to tag each 
training cycle with its own name

##config.py

This file collates all command line arguments that might be called with any of the files in ../tools. I store them all in one file here
to avoid circular imports within the files in ../tools.
