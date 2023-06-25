# CodeForKDD23

Here is the code and hyperparameter settings for the paper *Efficient and Joint Hyperparameter and Architecture Search for Collaborative Filtering, KDD 2023*.

## Optimal hyperparameters

| Hyperparameter/Dataset | MovieLens-100K | MovieLens-1M |
| ---------------------- | -------------- | ------------ |
| optimizer              | Adam           | Adam         |
| learning rate          | 0.0023         | 0.0014       |
| Embedding dimension    | 64             | 32           |
| Weigh decay            | 0.001          | 0.001        |
| Batch size             | 2000           | 1000         |



# Hyperparameter settings

| Hyperparameter                              |      |
| ------------------------------------------- | ---- |
| quantile in BORE                            | 0.25 |
| number of estimators in RF                  | 200  |
| Top-k: k                                    | 20   |
| number of training iterations               | 1000 |
| models sampled for hyperparameter screening | 30   |

