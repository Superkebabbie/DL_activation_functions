# DL_activation_functions

A small deep learning project to experiment with different techniques.

## Setup

First install `virtualenv` using: ``pip install virtualenv``.

Now move to the folder you would like to store your virtual environment in and type the following: ``virtualenv -p python3 DL_activation_functions``.

This will create a virtual `python3` environment called *DL_activation_functions*. 
You can activate a virtualenvironment using the following command: ``source path/to/virtualenv/DL_activation_functions``.

Your command line should now show the following:
```
(DL_activation_functions) me@myComputer: 
```

### Installing packages
Installing python packages still works the same:
``pip install _packagename_``.

However, we now also have a requirements file, which holds all the packages used in this project.
Whenever you install a new package you should add it to the requirements file.
This file can be found in the _DL_activation_functions_ project folder you cloned from this repository.

Update the `requirements.txt` file using: ``pip freeze > requirements.txt`` when in the project folder.
After this you should commit this new version of the file so others get the same packages as you do!

Install packages from the requirements file using: ``pip install -r requirements.txt``.