## Multi-Class Disk Distributions
An unofficial implementation of "Accurate Synthesis of Multi-Class Disk Distributions" [EG 2019]

### Structure
* ```pure_cpp```: forked from [here](https://github.com/Helios77760/ASMCDD), but remove all opengl stuff to a pure cpp version, super simple to run.
* ```simpler_cpp```: an even simpler version compared with ```pure_cpp```, still improving.
* ```python```: a python/pytorch implementation from scratch, still working

### How to run C++:
* Compile
```
cd simpler_cpp
mkdir build
cd build
cmake ..
make
```

* Run a single scene:
```
./DiskProject ../configs/constrained.txt
```

* Run all scenes
```
../scripts/run_all.sh
```


### How to run Python:
```
cd python
python main.py
```


### Reference
1. [code link](https://github.com/Helios77760/ASMCDD)
2. [paper link](https://hal.inria.fr/hal-02064699/file/Accurate_Synthesis_of_Multi_Class_Disk_Distributions%20%281%29.pdf)
