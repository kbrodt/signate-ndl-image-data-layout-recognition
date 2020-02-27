

# Signate: NDL Image Data Layout Recognition

[NDL Image Data Layout Recognition](https://signate.jp/competitions/218).

[4th place out of 97](https://signate.jp/competitions/218/leaderboard) (gold medal) with score 0.84310 (top1 -- 0.84978).


### Prerequisites

Install [`mmdetection`](https://github.com/open-mmlab/mmdetection).

```bash
git clone --branch v1.0.0 https://github.com/open-mmlab/mmdetection.git
```

Then

```bash
pip install -r requirements.txt
```

### Usage

First download the train and test data from the competition link.

To preprocess data, train, predict and ensamble run

```bash
bash ./run.sh
```

This will generates trained models and submission file.
