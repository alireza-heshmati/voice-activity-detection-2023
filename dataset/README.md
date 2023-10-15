# Dataset 


**Note1: For train, test, and evaluation of model, you should put unzipped data in [**Noisy_Dataset_V1**](./dataset/Noisy_Dataset_V1) and  [**Noisy_Dataset_V2**](./dataset/Noisy_Dataset_V2).**

**Note2: For running the codes, it is necessary to unzip test_filenames.zip, valid_filenames.zip, train_filenames_v2.zip, and VadLabel.zip files in this [**folder**](./dataset)**

**Note3: For each specific dataset, it is necessary to create train_filenames_v2.csv, test_filenames.csv, and valid_filenames.csv based on the path of your dataset in this [**folder**](./dataset).**

This [**folder**](./dataset) contains the project dataset for Voice Activity Detection.
<div align="justify"> In this project, around 4.6 millions mp3 files of data have been collected from noiseless Persian language and noise dataset. To create the noisy data, QUT dataset and the noisy data delivered from the employer were used as the pure noise dataset. The QUT dataset contains 5 different types of noise (home, cafe, street, etc.). 
To create VAD dataset, at first, a part of the Common Voice dataset was separated as bellow:
 
- Separating the validated section from the invalidated section.
- Separating the part that does not have a negative vote from the evaluation candidates and has at least one positive vote.
- Separating the part that is recognized by the internal model of the technical team for speech diagnosis completely.

After selecting the speech parts according to the above three criteria, we use the internal model of the technical team to segment the speech file. This model creates one character per frame of 20 milliseconds of input sound, and therefore we can assign speech and non-speech label to each frame. The data created by the above procedure constitutes the noise-free part of the DS-Fa-V01 and DS-Fa-V03 dataset. In this project, post-processing can be used to repair the gap between speech labels.


As a result, the noise-free part is gained from the DS-Fa-V02 dataset. To create noise samples in both dataset, the real QUT noise data and noise data created by the respected employer were used. For this reason, the noise-free part of two dataset DS-Fa-V01, DS-Fa-V02 and DS-Fa-V03 with different SNRs from **-2**to **30**  dB with **2** steps with QUT data and from 0 to 30 dB with 10 steps with The employer-created noise was mixed. For DS-Fa-V03, we added extra employer-created noise levels such as 6, 8, 12, 14, ..., 18 and 25 dB to the noise-free part to rich the dataset. For each level (SNR), the entire VAD dataset was combined with the noise that was chosen randomly during mixing. The last version of the dataset (DS-Fa-V03) containing around 4.6 millions of labeled data. For segmenting the dataset into evaluation and test data, around 160 thousands of DS-Fa-V01 data, were separated and prepared with the corresponding noise data of evaluation and test data. The others were classified as train data. Therefore, there is no leakage of the audio files of the training set in the evaluation and test parts.


```
DS-Fa-V03
	├── -2dB 
	│   ├── common_voice_fa_18202356SPLITCAR-WINUPB-1SPLIT0dB.mp3
	│   ├── common_voice_fa_18202357SPLITCAFE-CAFE-1SPLIT0dB.mp3
	│   ├── common_voice_fa_18202375SPLITREVERB-CARPARK-2SPLIT0dB.mp3	 	 
	│   ├──    .
	│   ├──    .	 	 
	│   └── common_voice_fa_18202378SPLITCAFE-CAFE-1SPLIT0dB.mp3
	├── 0 dB 
	│   ├── common_voice_fa_18202356SPLITCAR-WINUPB-1SPLIT0dB.mp3
	│   ├── common_voice_fa_18202357SPLITCAFE-CAFE-1SPLIT0dB.mp3
	│   ├── common_voice_fa_18202375SPLITREVERB-CARPARK-2SPLIT0dB.mp3	 	 
	│   ├──    .
	│   ├──    .	 	 
	│   └── common_voice_fa_18202378SPLITCAFE-CAFE-1SPLIT0dB.mp3
        .
        .
        .
        .
	├── 30 dB 
	│   ├── common_voice_fa_18202356SPLITCAR-WINUPB-1SPLIT0dB.mp3
	│   ├── common_voice_fa_18202357SPLITCAFE-CAFE-1SPLIT0dB.mp3
	│   ├── common_voice_fa_18202375SPLITREVERB-CARPARK-2SPLIT0dB.mp3	 	 
	│   ├──    .
	│   ├──    .	 	 
	│   └── common_voice_fa_18202378SPLITCAFE-CAFE-1SPLIT0dB.mp3
	└── InfdB
	    ├── common_voice_fa_18202356SPLITCAR-WINUPB-1SPLIT0dB.mp3
	    ├── common_voice_fa_18202357SPLITCAFE-CAFE-1SPLIT0dB.mp3
	    ├── common_voice_fa_18202375SPLITREVERB-CARPARK-2SPLIT0dB.mp3	 	 
	    ├──    .
	    ├──    .	 	 
	    └── common_voice_fa_18202378SPLITCAFE-CAFE-1SPLIT0dB.mp3

```
## Data collection procedure

In this project,the CommonVoice Persian version 13 database has been used to build a proper VAD database in Persian language.
CommonVoice is an open source project started by Mozilla to collect speech data, where people can speak sentences.

```bibtex
@article{nezami2019shemo,
  title={ShEMO: a large-scale validated database for Persian speech emotion detection},
  author={Nezami, Omid Mohamad and Lou, Paria Jamshid and Karami, Mansoureh},
  journal={Language Resources and Evaluation},
  volume={53},
  number={1},
  pages={1--16},
  year={2019},
  publisher={Springer}
}
```