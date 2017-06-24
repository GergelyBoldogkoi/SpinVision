SPIN VISION

Installation: 
This section is going to give a short explanation of how to use the code and
how to install the required software. First of all you are going to need to download jAER, the
software from inilabs that allows your computer to communicate with the DVS camera and to
make recordings. Instructions can be found on the inilabs website (5) . Then install python 2.7 on
your computer, although the code should be compatible with Python 3 as well. Before installing
spynnaker make sure you have installed all dependencies (6) . Then, go on to install the software
necessary for SpiNNaker (7) . Then install ’paer’ library (8) . Here take care to use the commit before
the final one, as the freshest commit breaks the entire library (Not the work of the author of this
project)! Now you are ready to clone the github repository (9) containing this project.


Running the code:
Before attempting to run any code make sure that you are connected to a
SpiNNaker machine. The code as well as most tests are going to fail if you are not. The training.py
file contains everything you need to get an overview of how to train the network. You should do
your experimentation by running code from that file, or alternatively you can create your own, but
make sure to include all the import statements. Most of this file consists of defining the locations
of the training and test files. When training or evaluating a network you always have to supply
the size of the input layer.



5 https://inilabs.com/support/software/jaer/#h.m0oac0618b3y

6 http://spinnakermanchester.github.io/common_pages/3.0.0/PythonInstall.html

7 http://spinnakermanchester.github.io/spynnaker/3.0.0/PyNNOnSpinnakerInstall.html

8 https://github.com/bio-modelling/py-aer

9 https://github.com/GergelyBoldogkoi/SpinVision



MATLAB CODE
The Simulink Animation compatible with MATLAB R2016a can be found in the Ball Simulation Folder.
