# E2SRLF: End-to-End Light Field Image Super-Resolution Depth Estimation


## Requirement

- PyTorch >= 1.13.0, torchvision >= 0.15.0. The code is tested with python=3.8, cuda=11.0.
- A GPU with enough memory.
- The disparity range is [-4, 4], which means our dilation rate is set to 9.


## Datasets

- We used the HCI 4D LF benchmark for training and evaluation. Please refer to the [benchmark website](https://lightfield-analysis.uni-konstanz.de/) for details.
- 

### Path structure
- If not specified, it is a file name; if specified, it is a general term for multiple files.
```
.
├── dataset
│   ├── training
│   └── validation
├── Figure
│   └── paper_picture
├── implement 
│   ├── implementation     # This is a general term for multiple files.
│   └── data_preprocessing # This is a general term for multiple files.
├── model
│   └── network_functions  # This is a general term for multiple files.
├── param
│   └── checkpoints 
└── Results
    ├── our_network        # This is a general term for multiple files.
    │   ├── E2SRLF
    │   └── E2SRLF_x1
    └── Analysis           # This is a general term for multiple files.
        ├── E2SRLF_NAT 
        ├── E2SRLF_SACAT
        └── E2SRLF_SRL1

```

### Train

- Set the hyper-parameters in parse_args() if needed. We have provided our default settings in the realeased codes.
- You can train the network by calling implement.py and giving the mode attribute to train.  
    ``` python implement/implement.py --net Net_Full  --n_epochs 3000 --mode train --device cuda:1 ```

- Checkpoint will be saved to ./param/'NetName'.
  
### Valition and Test

- After loading the weight file used by your domain, you can call implement.py and giving the mode attribute to valid or test.
- The result files (i.e., scene_name.pfm) will be saved to ./Results/'NetName'.

### Results

#### Contrast with the state-of-the-art work



