You can `git clone` the data from [here](https://huggingface.co/datasets/a2d8a4v/SpeechOcean762_for_ConPCO/tree/main). For the GOP features and targets, we use them from the `data` in [GOPT](https://github.com/YuanGongND/gopt) (Gong et al., 2022) GitHub repository. After integration, the directory structure should be as follows:

```
.
├── README.md
└── seq_data_librispeech_v4
    ├── te_dur_feat.npy
    ├── te_energy_feat.npy
    ├── te_feat.npy
    ├── te_hubert_feat_v2.npy
    ├── te_label_phn.npy
    ├── te_label_utt.npy
    ├── te_label_word.npy
    ├── te_w2v_300m_feat_v2.npy
    ├── te_wavlm_feat_v2.npy
    ├── te_word_id.npy
    ├── tr_dur_feat.npy
    ├── tr_energy_feat.npy
    ├── tr_feat.npy
    ├── tr_hubert_feat_v2.npy
    ├── tr_label_phn.npy
    ├── tr_label_utt.npy
    ├── tr_label_word.npy
    ├── tr_w2v_300m_feat_v2.npy
    ├── tr_wavlm_feat_v2.npy
    └── tr_word_id.npy
```

For details, the following describes how the preprocess procedure works to obtain these features.

 - GOP (Goodness-of-Pronunciation) Features

You can obtain the GOP (Goodness-of-Pronunciation) features by following the [gop_speechocean762](https://github.com/kaldi-asr/kaldi/tree/master/egs/gop_speechocean762) recipe in the Kaldi project, then padding them to a maximum length of 50. For a comprehensive understanding, refer to the [GOPT](https://github.com/YuanGongND/gopt) (Gong et al., 2022) GitHub repository. In this work, we directly use the GOP features from the `data` in [GOPT](https://github.com/YuanGongND/gopt) (Gong et al., 2022) GitHub repository.

- Energy Features

For the energy component, we utilize root-mean-square energy with seven statistical dimensions: mean, standard deviation, median, median absolute deviation, maximum, minimum, and summation.

- Duration Features

Duration is retrieved for each phone in the phoneme sequences after performing phone forced alignment. For more details, refer to the [gop_speechocean762](https://github.com/kaldi-asr/kaldi/tree/master/egs/gop_speechocean762) recipe in the Kaldi project, which outlines the process for computing GOP features.

 - Self-Supervised Learning Audio Features (e.g., Wav2vec 2.0, HuBERT, WavLM)

SSL embeddings are obtained by chunking based on phone durations and computing the mean of the resulting segments.

- Word IDs Features

To generate word embeddings, words are converted to word IDs, excluding special tokens such as `['<eps>', '!SIL', '<SPOKEN_NOISE>', '<UNK>', '#0', '<s>', '</s>']`. After generating the lexicon graph, the pre process creates a `word.txt` file in the `dict` directory. For further details, refer to the generated lexicon graph documentation.

- Targets

To obtain target data, refer to the `data` in [GOPT](https://github.com/YuanGongND/gopt) (Gong et al., 2022) GitHub repository, download those targets, and integrate it with the `data` in this repository.
