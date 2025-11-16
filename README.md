
# PyTorch Implementation of RFxG


## Introduction
This is the official PyTorch implementation of the Reference-Frame x Granularity (RFxG) taxonomy.

Saliency maps have become a cornerstone of visual explanation in deep learning, yet there remains no consensus on their intended purpose and their alignment with specific user queries. This fundamental ambiguity undermines both the evaluation and practical utility of explanation methods. In this paper, we introduce the Reference-Frame x Granularity (RFxG) taxonomy—a principled framework that addresses this ambiguity by conceptualizing saliency explanations along two essential axes: the reference-frame axis (distinguishing between pointwise "Why Husky?" and contrastive "Why Husky and not Shih-tzu?" explanations) and the granularity axis (ranging from fine-grained class-level to coarse-grained group-level interpretations, e.g., “Why Husky?” vs. “Why Dog?”). Through this lens, we identify critical limitations in existing evaluation metrics, which predominantly focus on pointwise faithfulness while neglecting contrastive reasoning and semantic granularity. To address these gaps, we propose four novel faithfulness metrics that systematically assess explanation quality across both RFxG dimensions. Our comprehensive evaluation framework spans ten state-of-the-art methods, four model architectures, three datasets, and targeted user studies. By suggesting a shift from model-centric to user-intent-driven evaluation, our work provides both the conceptual foundation and practical tools necessary for developing explanations that are not only faithful to model behavior but also meaningfully aligned with human understanding.

## Producing Classification Saliency Maps
Images should be stored in the `data\ILSVRC2012_img_val` directory. 
The information on the images being evaluated and their target class can be found in the `data\pics.txt` file, with each line formatted as follows: `<file_name> target_class_number` (e.g. `ILSVRC2012_val_00002214.JPEG 153`).

To generate saliency maps, run the following command:
```
python rfxg_xai.py
```


By default, saliency maps for CNNs are generated using the Resnet101 network on the last layer. You can change the selected network and layer by modifying the `model_name` and `FEATURE_LAYER_NUMBER` variables in the `rfxg_xai.py` class. For ViT, the default is ViT-Base, which can also be configured using the `model_name` variable.

The generated saliency maps will be stored in the `qualitive_results` directory.
### ViT models weight files:
- ViT-B [Link to download](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)
- ViT-S [Link to download](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth)


## Credits
For comparison, we used the following implementations of code from git repositories:
- https://github.com/jacobgil/pytorch-grad-cam
- https://github.com/pytorch/captum
- https://github.com/PAIR-code/saliency



