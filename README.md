## Multi-Class Disk Distributions
An unofficial implementation of "Accurate Synthesis of Multi-Class Disk Distributions" [EG 2019]

### Structure
* ```pure_cpp```: forked from [here](https://github.com/Helios77760/ASMCDD), but remove all opengl stuff to a pure cpp version, super simple to run.
* ```simpler_cpp```: an even simpler version compared with ```pure_cpp```, still improving.
* ```python```: a python/pytorch implementation from scratch, still working

### How run C++:
```
cd pure_cpp
mkdir build
cd build
cmake ..
make
./DiskProject ../configs/constrained.txt
```

### How run Python:
```
cd python
python main.py
```


### Acknowledgements
* Reference: [here] https://github.com/Helios77760/ASMCDD

