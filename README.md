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
  - dataset: 
      - dog
      - cat
  - python split_dataset.py --path-dataset PathDataset --path-save PathSave
- else:
  - dataset:
    - train
        - dog
        - cat
    - val
        - dog
        - cat
    - test
        - dog
        - cat
## Training
- change numbers classes with variable num_classes in file train at line 18
  - from models.MobilenetV2 import mobilenet_v2
  - model = mobilenet_v2(pretrained=True, num_classes=2).to(device)
- run training model
  - python train.py --dataset PathDataset --epochs NumbersEpochs --batch-size SizeOfBatch --image-size SizeInput
## Test
- python test.py --file-folder FolderContainImage --folder-model FolderContainModel
## Evaluate model
- construct test folder example:
  - test
    - dog
    - cat
- python evaluate_model.py --folder-test FolderTest --folder-model FolderContainModel --path-save FolderSave
## Reference
- Release of advanced design of MobilenetV2 [ICCV2019](https://arxiv.org/pdf/1801.04381.pdf)
- Release of advanced pre-trained MobilenetV2 imagenet [Pytoch](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
