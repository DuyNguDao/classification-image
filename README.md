# Backbone of MobilenetV2 with Pytorch
This is backbone use pretraining model deep learning with network mobilenetV2 and evaluate model
## Dev
- Dao Duy Ngu
- Le Van Thien
## Install
### Anaconda
- conda create --name deep python=3.8
- conda activate deep
### Packages
- pip install -r requirements.txt
## Construct dataset
Example: Classification dog and cat
- if dataset not split
  - dataset: dog and cat
  - python split_dataset.py --path-dataset <dataset> --path-save <dataset_split>
- else:
  - dataset: train, val, test
    - train, val, test folder have two folder cat and dog
## Training
- change numbers classes with variable num_classes in file train at line 18
  - from models.MobilenetV2 import mobilenet_v2
  - model = mobilenet_v2(pretrained=True, num_classes=2).to(device)
- run training model
  - python train.py --dataset <path dataset> --epochs <numbers epochs> --batch-size <size of batch> --image-size <size input>
## Test
- python test.py --file-folder <folder contain image> --folder-model <folder contain model>
## Evaluate model
- construct test folder example
- test
  - dog
  - cat
- python evaluate_model.py --folder-test <folder test> --folder-model <folder contain model> --path-save <folder save>
## Reference
- Release of advanced design of MobilenetV2 [ICCV2019](https://arxiv.org/pdf/1801.04381.pdf)
- Release of advanced pre-trained MobilenetV2 imagenet [Pytoch](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)














