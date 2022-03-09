# Foggy-CycleGAN

<p align="center">
 <img src="images/banner-cropped-rnd.png">
</p>

This project is the pre-processing procedure (rendering) for All Weather high-level task (Recognition).

## Description
**Foggy-CycleGAN** is a
<a href="https://junyanz.github.io/CycleGAN/" target="_blank">CycleGAN</a> model trained to synthesize fog on clear images.

## Pre-trained Models
A version of pre-trained models used in the thesis can be found [here](https://drive.google.com/drive/folders/1QKsiaGkMFvtGcp072IG57MfY1o_D-L3k?usp=sharing).

## Notebook <a href="https://colab.research.google.com/github/ghaiszaher/Foggy-CycleGAN/blob/master/Foggy_CycleGAN.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
A Jupyter Notebook file <a href="https://github.com/ghaiszaher/Foggy-CycleGAN/blob/master/Foggy_CycleGAN.ipynb" target="_blank">Foggy_CycleGAN.ipynb</a> is available in the repository.
 
## Using
To utilize this code and generate foggy images. You only have to run <a href="https://github.com/Blackpinkup/Foggy-CycleGAN/inference.py" target="_blank">inference.py</a>. Note that you have to modify your image path in inference.py.

## Results
<p align="center">
 <img src="000000000074.jpg">
</p>
<p align="center">
 origin image
</p>

<p align="center">
 <img src="00_intensity_0.25.jpg">
</p>
<p align="center">
 haze 0.25
</p>

<p align="center">
 <img src="01_intensity_0.50.jpg">
</p>
<p align="center">
 haze 0.5
</p>

<p align="center">
 <img src="02_intensity_0.75.jpg">
</p>
<p align="center">
 haze 0.75
</p>
