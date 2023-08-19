# Dataset

This [**folder**](../datasets) contains the project dataset for Voice Activity Detection.
<div align="justify"> In this project, 2400 hours of data have been collected from different languages ​​(English, German, French, Welsh, Turkish, and 13 other languages) of which 1900 hours have been approved (confirmed). To create the noise data, the QUT database and the noise data delivered from the employer were used as the pure noise database. The QUT database contains 5 different types of noise (home, cafe, street, etc.). 
To create VAD data at first, a part of the Common Voice database was separated as follows(bellow):


 
- Separating (isolating) the validated section from the invalidated section.
- Separating the part that does not have a negative vote from the evaluation candidates and has at least one positive vote.
- Separating the part that is recognized by the internal model of the technical team for speech diagnosis completely.

After selecting the speech parts according to the above three criteria, we use the internal model of the technical team to segment the speech file. This model creates one character per frame of 20 milliseconds of input sound, and therefore we can assign speech and non-speech labels to each frame. The data created by the above procedure constitutes the noise-free part of the DS-Fa-V01 database. If needed to conjunction between production speech labels, the disjunction (discontinuity) of less than 200 milliseconds in the speech tag can be removed using post-processing.


As a result, the noise-free part is gained from the DS-Fa-V02 database. To create noise samples in both databases, the real QUT noise data and noise data created by the respected employer were used. For this reason, the noise-free part of two databases DS-Fa-V01 and DS-Fa-V02 with different SNRs from **-2**to **30**  dB with **2** steps with QUT data and from 0 to 30 dB with 10 steps with The employer-created noise was mixed. For each level (SNR), the entire VAD database was combined with the noise that was chosen randomly during mixing. Our database containing about 3.5 million labeled data was prepared, which were named DS-Fa-V01 and DS-Fa-V02. For segmenting the database into evaluation and test data, at first, 160 thousand pure data of each part that contains 10% of the total files, were separated and prepared with the corresponding noise data of evaluation and test data. The other data, which make up 80% of the total, were classified as education data. Therefore, there is no leakage of the audio files of the training set in the evaluation and test parts.

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




