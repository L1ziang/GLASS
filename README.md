# _GAN you see me? enhanced data reconstruction attacks against split inference_ - NeurIPS 2023

## 📄 [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html)

## 📝 Abstract
_Split Inference (SI) is an emerging deep learning paradigm that addresses computational constraints on edge devices and preserves data privacy through collaborative edge-cloud approaches. However, SI is vulnerable to Data Reconstruction Attacks (DRA), which aim to reconstruct users' private prediction instances. Existing attack methods suffer from various limitations. Optimization-based DRAs do not leverage public data effectively, while Learning-based DRAs depend heavily on auxiliary data quantity and distribution similarity. Consequently, these approaches yield unsatisfactory attack results and are sensitive to defense mechanisms. To overcome these challenges, we propose a **G**AN-based **LA**tent **S**pace **S**earch attack (**GLASS**) that harnesses abundant prior knowledge from public data using advanced StyleGAN technologies. Additionally, we introduce **GLASS++** to enhance reconstruction stability. Our approach represents the first GAN-based DRA against SI, and extensive evaluation across different split points and adversary setups demonstrates its state-of-the-art performance. Moreover, we thoroughly examine seven defense mechanisms, highlighting our method's capability to reveal private information even in the presence of these defenses._

## 📦 Environment Installation
```bash
pip install -r requirements.txt
```

## 🔨 Experiments
Please refer to ./celeba/README.md and ./cifar/cinic-10/README.md

## ⚙️ GAN Model Training
Please refer to ./stylegan2-pytorch-master_mlp8_mixing=0.9

## 🔥 Acknowledgement
[StyleGAN](https://github.com/rosinality/stylegan2-pytorch)

[StyleGAN-XL](https://github.com/autonomousvision/stylegan-xl)