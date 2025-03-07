# Lungs and BRaTs Datasets:
We intergrate our model into [UNETR++](https://tinyurl.com/2p87x5xn), the specific overflow as follows:
## Installation
### 1. Create and activate conda environment
```bash
conda create --name SIPL python=3.9
conda activate SIPL
```
### 2. Install PyTorch and torchvision
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
### 3. Install other dependencies
```bash
pip install -r requirements.txt
```
<hr />


## Dataset
### Lungs and Tumor Datasets
We follow the same dataset preprocessing as in [UNETR++](https://tinyurl.com/2p87x5xn). We conducted extensive experiments on two dataset: BRaTs and Decathlon-Lung. 

  The dataset folders for Decathlon-Lung should be organized as follows: 

```
./DATASET_Lungs/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task06_Lung/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task006_Lung
       ├── unetr_pp_cropped_data/
           ├── Task006_Lung
 ```
   The dataset folders for BRaTs should be organized as follows: 

```
./DATASET_Tumor/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task03_tumor/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task003_tumor
       ├── unetr_pp_cropped_data/
           ├── Task003_tumor
 ```
 
Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details.
Alternatively, you can download the preprocessed dataset for [Decathlon-Lung](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EWhU1T7c-mNKgkS2PQjFwP0B810LCiX3D2CvCES2pHDVSg?e=OqcIW3), [BRaTs](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/EaQOxpD2yE5Btl-UEBAbQa0BYFBCL4J2Ph-VF_sqZlBPSQ?e=DFY41h), and extract it under the project directory.

## Training
The following scripts can be used for training our [UNETR++](https://github.com/Amshaker/unetr_plus_plus) model on the datasets:
```shell
bash training_scripts/run_training_lung.sh
bash training_scripts/run_training_tumor.sh
```
<hr />

## Evaluation

To reproduce the results of UNETR++: 

1- Paste the trained model```model_final_checkpoint.model``` into the following path:
```shell
unetr_pp/evaluation/unetr_pp_lung_checkpoint/unetr_pp/3d_fullres/Task006_Lung/unetr_pp_trainer_lung__unetr_pp_Plansv2.1/fold_0/
```
Then, run 
```shell
bash evaluation_scripts/run_evaluation_lung.sh
```
Enter into the path ```unetr_pp/evaluation/unetr_pp_lung_checkpoint/unetr_pp/3d_fullres/Task006_lung/unetr_pp_trainer_lung__unetr_pp_Plansv2.1/```, run
```bash
python cal_dice_score.py
```

2- Paste the trained model ```model_final_checkpoint.model``` into the following path:
```shell
unetr_pp/evaluation/unetr_pp_tumor_checkpoint/unetr_pp/3d_fullres/Task003_tumor/unetr_pp_trainer_tumor__unetr_pp_Plansv2.1/fold_0/
```
Then, run 
```bash
bash evaluation_scripts/run_evaluation_tumor.sh
```
Enter into the path ```unetr_pp/evaluation/unetr_pp_tumor_checkpoint/unetr_pp/3d_fullres/Task003_tumor/unetr_pp_trainer_tumor__unetr_pp_Plansv2.1/```, run
```bash
python cal_dice_score.py
```
<hr />
