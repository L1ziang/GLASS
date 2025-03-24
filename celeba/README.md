## 🔍 Datasets & GAN models
- [./data](https://pan.baidu.com/s/1KEHREObxO19KtICAEfJeew?pwd=6qkj)
- [./models](https://pan.baidu.com/s/1xOwdoUQPrODnCVjEqzc7iQ?pwd=ehfc)

## 🗡️ Attacks

- **rMLE** [Model inversion attacks against collaborative inference](https://dl.acm.org/doi/abs/10.1145/3359789.3359824)
```bash
python optimize_onlyX.py --config_file config_onlyX.yaml --stage onlyX --index 1
```

- **LM** [DISCO: Dynamic and Invariant Sensitive Channel Obfuscation for Deep Neural Networks](https://openaccess.thecvf.com/content/CVPR2021/html/Singh_DISCO_Dynamic_and_Invariant_Sensitive_Channel_Obfuscation_for_Deep_Neural_CVPR_2021_paper.html)
```bash
python optimize_onlyM.py --config_file config_onlyM.yaml --stage onlyX --index 2
```

- **IN** [Model inversion attacks against collaborative inference](https://dl.acm.org/doi/abs/10.1145/3359789.3359824)
```bash
python inverse_attack.py --config_file config_inv.yaml --stage inversion --index 3
```

- **GLASS** [GAN you see me? enhanced data reconstruction attacks against split inference](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html)
```bash
python optimize_Gan_adam_z_w+.py --config_file config_Gan_zAndw+.yaml --stage Gan --index 4
```

- **GLASS++** [GAN you see me? enhanced data reconstruction attacks against split inference](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ab003a4f85ecb1b7b1514ff539dc7395-Abstract-Conference.html)
```bash
python learning_PSP_inversion.py --config_file config_PSP.yaml --stage inversion --index 5
```
```bash
python optimize_Gan_w+_encoderbase.py --config_file config_PSP.yaml --stage Gan --index 6
```

## 🛡️ Defenses

- **Dropout Defense** [Model inversion attacks against collaborative inference](https://dl.acm.org/doi/abs/10.1145/3359789.3359824)
```bash
python dropout_defense.py --config_file config_dropout.yaml --stage noisytrain --index 7
```

- **DISCO** [DISCO: Dynamic and Invariant Sensitive Channel Obfuscation for Deep Neural Networks](https://openaccess.thecvf.com/content/CVPR2021/html/Singh_DISCO_Dynamic_and_Invariant_Sensitive_Channel_Obfuscation_for_Deep_Neural_CVPR_2021_paper.html)
```bash
python disco_defense.py --config_file config_disco.yaml --stage noisytrain --index 8
```

- **Noise Mask** [Practical defences against model inversion attacks for split neural networks](https://arxiv.org/abs/2104.05743)
```bash
python noise_defense.py --config_file config_noise.yaml --stage noisytrain --index 9
```

- **Shredder** [Shredder: Learning noise distributions to protect inference privacy](https://dl.acm.org/doi/abs/10.1145/3373376.3378522)
```bash
python cloak_defense.py --config_file config_cloak.yaml --stage noisytrain --index 10
```

- **Adversarial Learning** [DeepObfuscator: Obfuscating intermediate representations with privacy-preserving adversarial learning on smartphones](https://dl.acm.org/doi/abs/10.1145/3450268.3453519)
```bash
python adv_defense.py --config_file config_adv.yaml --stage advtrain --index 11
```

- **NoPeek** [NoPeek: Information leakage reduction to share activations in distributed deep learning](https://ieeexplore.ieee.org/abstract/document/9346367)
```bash
python NoPeek_defense.py --config_file config_NoPeek.yaml --stage noisytrain --index 12
```

- **Siamese Defense+** [A hybrid deep learning architecture for privacy-preserving mobile analytics](https://ieeexplore.ieee.org/abstract/document/8962332)
```bash
python siamese_defense.py --config_file config_siamese.yaml --stage noisytrain --index 13
```





