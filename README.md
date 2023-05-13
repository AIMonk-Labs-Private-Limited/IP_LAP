# IP_LAP: Identity-Preserving Talking Face Generation with Landmark and Appearance Priors （CVPR 2023）

Pytorch official implementation for our CVPR2023 paper "****I****dentity-****P****reserving Talking Face Generation with ****L****andmark and ****A****ppearance ****P****riors".

<img src='./CVPR2023framework.png' width=900>

TODO:(****will finish before May 15th****)
- [x] Demo videos
- [x] pre-trained model
- [x] code for testing
- [x] code for training
- [x] code for preprocess dataset
- [ ] guidline 
- [ ] arxiv paper release

[[Paper]](https://arxiv.org/abs/coming_soon) [[Demo Video]](https://youtu.be/wtb689iTJC8)

## Requirements
- Python 3.7.13
- torch 1.10.0
- torchvision 0.11.0
- ffmpeg

We conduct the experiments with 4 24G RTX3090 on CUDA 11.1. For more details, please refer to the `requirements.txt`. We recommand to install [pytorch](https://pytorch.org/) firsrly, and then run:
```
pip install -r requirements.txt
```
## Test
Download the pre-trained models from [oneDrive](https://1drv.ms/f/s!Amqu9u09qiUGi7UJIADzCCC9rThkpQ?e=P1jG5N) or [jianguoyun](https://www.jianguoyun.com/p/DeXpK34QgZ-EChjI9YcFIAA), and place them to the folder `test/checkpoints` . Then run the following command:
```
CUDA_VISIBLE_DEVICES=0 python inference_single.py
```
To inference on other videos, please specify the `--input` and `--audio` option and see more details in code.

## Train
### download LRS2 dataset
Our models are trained on LRS2. Please go to the [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) website to download the dataset. LRS2 dataset folder structure is following:
```
data_root (mvlrs_v1)
├── main, pretrain (we use only main folder in this work)
|	├── list of folders
|	│   ├── five-digit numbered video IDs ending with (.mp4)
```
`main folder` is the `lrs2_video` mentioned below.

### preprocess the audio
firstly we run 
```
CUDA_VISIBLE_DEVICES=0 python preprocess_audio.py --data_root ...../lrs2_video/ --out_root ./lrs2_audio
```
### preprocess the videos face 

```
UDA_VISIBLE_DEVICES=0 python preprocess_video.py --dataset_video_root ....../lrs2_video/
```

### Train Landmark generator
```
CUDA_VISIBLE_DEVICES=0 python train_landmarks_generator.py --pre_audio_root ./lrs2_audio/ --landmarks_root ./lrs2_landmarks/
```

### Train video renderer
run the 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_video_renderer.py --sketch_root ...../lrs2_sketch/ --face_img_root ..../lrs2_face/  --audio_root ...../lrs2_audio/
```
note：the translation module will only be trained  after 25 epoches, thus the fid and running_gen_loss will only decrease after epoch 25. 


## Acknowledgement
This project is built upon the publicly available code [DFRF](https://github.com/sstzal/DFRF) and [Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/master). Thanks the authors of DFRF and Wav2Lip for making their excellent work and codes publicly available.




## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{zhong2023identity-preserving,
  title={Identity-Preserving Talking Face Generation with Landmark and Appearance Priors},
  author={Weizhi Zhong, Chaowei Fang, Yinqi Cai, Pengxu Wei, Gangming Zhao, Liang Lin, Guanbin Li},
  booktitle="Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition",
  year={2023}
}
```



