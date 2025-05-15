

# Counterfactual Knowledge Maintenance for Unsupervised Domain Adaptation

This repository contains the code for 'Counterfactual Knowledge Maintenance for Unsupervised Domain Adaptation'

---
<div align="center">
  <img src="assets/overall.pdf" width="900px" />
</div>
---
 
## How to Install Dependent Environments
Our code is built based on CLIP and Dassl, which can be installed with following commands.

```sh

# install CLIP

pip install git+https://github.com/openai/CLIP.git


# install Dassl

git clone https://github.com/KaiyangZhou/Dassl.pytorch.git

cd dassl

pip install -r requirements.txt

pip install .

cd..

```
One can install other dependent tools via

```sh
pip install -r requirements.txt
```
## How to Download Datasets
The datasets used for UDA tasks can be downloaded via the following links.

VisDA17 (http://ai.bu.edu/visda-2017/#download)

Office-Home (https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)

Mini-DomainNet (http://ai.bu.edu/DomainNet/)

After downloading the datasets, please update the dataset paths in `scripts/{dataset}.sh` accordingly.

## How to Run the Code

We provide scripts for running UDA experiments on Office-Home, VisDA17, Mini-DomainNet datasets in the `scripts` folder.

For instance, to run a task on Office-Home:

```bash

cd scripts

sh office_home.sh

```

## Citation
If you find the code useful in your research, please consider citing:

    


## Acknowledgments

This project builds upon the invaluable contributions of following open-source projects:

1. DAMP (https://github.com/TL-UESTC/DAMP)
2. AD-CLIP (https://github.com/mainaksingha01/AD-CLIP)





