## ⚙️ Target model training
```bash
python vanilla_main.py --config_file config_pretrain.yaml --stage pretrain
```

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
python optimize_Gan_cma_w.py --config_file config_Gan_w_cma.yaml --stage Gan --index 4
```