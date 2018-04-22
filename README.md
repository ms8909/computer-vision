# S³FD: Single Shot Scale-invariant Face Detector
A PyTorch Implementation of Single Shot Scale-invariant Face Detector converted to Face Recognition system.

Testing and Eval code at : https://github.com/clcarwin/SFD_pytorch

## Train 
python3 training.py

## Model
[s3fd_convert.7z](https://github.com/clcarwin/SFD_pytorch/releases/tag/v0.1)

## Test
```
python test.py --model data/s3fd_convert.pth --path data/test01.jpg
```
![output](data/test01_output.png)

# References
[SFD](https://github.com/sfzhang15/SFD)
