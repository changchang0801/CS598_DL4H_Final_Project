
Paper Reproduction Study - Patient Outcome Prediction with TPC Networks
===============================================


This repository contains the code used for replicating paper **Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit** (published at **ACM CHIL 2021**) and implementation instructions. 


## Motivation
It is a central task in hospital operation and resources planning to reliably predict the length of stay for hospitalized patients, especially for those critically ill patients staying in the Intensive Care Unit (ICU). Our selected paper aims to build a novel and reliable deep-learning predictive model for the length of stay in the ICU.

Historically, the two most popular models used for the length of stay prediction are LSTMs and Transformers due to the centrality of time series in the EHR. More recently Temporal Convolutional Networks (TCN) were developed as a variation of CNN to handle sequential data. Our selected paper described a novel Temporal Pointwise Convolution (TPC) model, which is based on the combination of temporal convolution networks and pointwise (1x1) convolution to predict ICU length of stay. The TPC model is specifically designed to handle common challenges with EHR, such as skewness, irregular sampling, and missing data. In the paper, the TPC model significantly outperforms the commonly used LSTM and Transformer models for ICU length of stay prediction.
## Headline Results

### Length of Stay Prediction

We report on the following metrics: 
- Mean absolute deviation (MAD)
- Mean absolute percentage error (MAPE)
- Mean squared error (MSE)
- Mean squared log error (MSLE)
- Coefficient of determination (R<sup>2</sup>)
- Cohen Kappa Score (Harutyunyan et al. 2019)

For the first four metrics, lower is better. For the last two, higher is better.


**Comparison with results from the paper**

| Source          | Model       | MAD        | MAPE      | MSE       | MSLE     | R²        | Kappa    |
|-----------------|-------------|------------|-----------|-----------|----------|-----------|----------|
| Paper           | TPC         | 1.95±0.02  | 72.0±3.1  | 23.8±0.4  | 0.87±0.03| 0.19±0.01 | 0.51±0.01|
|                 | CW-LSTM     | 2.40±0.01  | 123.4±0.7 | 26.5±0.1  | 1.48±0.01| 0.1±0.0   | 0.31±0.00|
|                 | LSTM        | 2.41±0.01  | 129.9±1.9 | 26.2±0.2  | 1.52±0.00| 0.11±0.1  | 0.31±0.01|
|                 | Transformer | 2.39±0.00  | 120.1±0.6 | 26.5±0.1  | 1.48±0.00| 0.10±0.0  | 0.31±0.00|
| --------------- |-------------|------|--------|-------|------|------|-------|
| Replication     | TPC         | 1.74       | 41.86     | 22.02     | 0.47     | 0.37      | 0.67     |
|                 | CW-LSTM     | 2.68       | 123.54    | 32.25     | 1.49     | 0.08      | 0.30     |
|                 | LSTM        | 2.67       | 133.17    | 31.46     | 1.54     | 0.10      | 0.32     |
|                 | Transformer | 2.65       | 117.45    | 32.19     | 1.47     | 0.08      | 0.30     |

All models are trained on 50% of training set. All evaluation metrics are calculated on testing set. Our results are largely consistent with those from paper, with TPC model outperforms the channelwise-LSTM (CW-LSTM), LSTM and Transformer models for ICU length of stay prediction by margins of 35\%. Consistent with the paper, other than TCP the best performing model is Transformer.

**Ablation studies results**

| Model         | MAD  | MAPE  | MSE    | MSLE | R²   | Kappa |
|---------------|------|-------|--------|------|------|-------|
| TPC           | 1.74 | 41.86 | 22.02  | 0.47 | 0.37 | 0.67  |
| Point. Only   | 2.77 | 30.98 | 142.71 | 1.53 | 0.12 | 0.40  |
| Temp. Only    | 1.85 | 53.97 | 23.30  | 0.62 | 0.34 | 0.64  |
| TPC(no skip)  | 1.87 | 59.69 | 22.63  | 0.68 | 0.36 | 0.63  |
| TPC(no diag)  | 1.66 | 43.15 | 21.05  | 0.45 | 0.40 | 0.70  |

The temporal-only model is superior to the pointwise-only model, but neither reaches the performance of the TPC model. Removing the skip connections reduces performance by 7%. Interestingly and surprisingly, we found that the exclusion of diagnoses does not seem to harm the mode. All above findings are described in the original paper.

**Comparison when using MSLE vs MSE**

| Loss function | Model       | MAD  | MAPE   | MSE   | MSLE | R²   | Kappa |
|---------------|-------------|------|--------|-------|------|------|-------|
| MSLE          | TPC         | 1.74 | 41.86  | 22.02 | 0.47 | 0.37 | 0.67  |
|               | CW-LSTM     | 2.68 | 123.54 | 32.25 | 1.49 | 0.08 | 0.30  |
|               | LSTM        | 2.67 | 133.17 | 31.46 | 1.54 | 0.10 | 0.32  |
|               | Transformer | 2.65 | 117.45 | 32.19 | 1.47 | 0.08 | 0.30  |
|---------------|-------------|------|--------|-------|------|------|-------|
| MSE           | TPC         | 2.21 | 111.13 | 22.13 | 1.90 | 0.37 | 0.55  |
|               | CW-LSTM     | 2.83 | 211.11 | 30.72 | 1.87 | 0.12 | 0.28  |
|               | LSTM        | 2.93 | 249.57 | 30.73 | 2.06 | 0.12 | 0.25  |
|               | Transformer | 2.89 | 241.17 | 30.36 | 2.00 | 0.13 | 0.26  |

Consistent with findings from the paper, we can see that using the MSLE (rather than MSE) loss function leads to significant improvements in all models, with large performance gains in MAD, MAPE, MSLE and Kappa.

**Comparison when excluding diagnosis code for all models**

| Contain diagnosis | Model       | MAD  | MAPE   | MSE   | MSLE | R²    | Kappa |
|-------------------|-------------|------|--------|-------|------|-------|-------|
| Yes               | TPC         | 1.74 | 41.86  | 22.02 | 0.47 | 0.37  | 0.67  |
|                   | CW-LSTM     | 2.68 | 123.54 | 32.25 | 1.49 | 0.08  | 0.30  |
|                   | LSTM        | 2.67 | 133.17 | 31.46 | 1.54 | 0.10  | 0.32  |
|                   | Transformer | 2.65 | 117.45 | 32.19 | 1.47 | 0.08  | 0.30  |
|-------------------|-------------|------|--------|-------|------|-------|-------|
| No                | TPC         | 1.66 | 43.15  | 21.05 | 0.45 | 0.40  | 0.70  |
|                   | CW-LSTM     | 2.65 | 128.53 | 32.20 | 1.48 | 0.08  | 0.30  |
|                   | LSTM        | 2.69 | 134.13 | 32.17 | 1.57 | 0.08  | 0.29  |
|                   | Transformer | 2.63 | 121.56 | 31.73 | 1.47 | 0.096 | 0.31  |


Intrigued by the surprising finding that the exclusion of diagnosis codes as input features actually improves TPC performance, we performed similar analysis to the rest of models without diagnosis codes as input. After skip diagnosis, rest of input variables include vital signs, nurse observations, machine logged variables among other time series data. In consistent with TPC model, we observed that exclusion of diagnosis does not seem to harm any models. In fact, there is minor performance improvements (about 1%) in all models but less than that seen in TPC model (7% improvement in MAD).


## Pre-processing Instructions

### eICU

1) To run the sql files you must have the eICU database set up: https://physionet.org/content/eicu-crd/2.0/. 

2) Follow the instructions: https://eicu-crd.mit.edu/tutorials/install_eicu_locally/ to ensure the correct connection configuration. 

3) Replace the eICU_path in `paths.json` to a convenient location in your computer, and do the same for `eICU_preprocessing/create_all_tables.sql` using find and replace for 
`'/Users/emmarocheteau/PycharmProjects/TPC-LoS-prediction/eICU_data/'`. Leave the extra '/' at the end.

4) In your terminal, navigate to the project directory, then type the following commands:

    ```
    psql 'dbname=eicu user=eicu options=--search_path=eicu'
    ```
    
    Inside the psql console:
    
    ```
    \i eICU_preprocessing/create_all_tables.sql
    ```
    
    This step might take a couple of hours.
    
    To quit the psql console:
    
    ```
    \q
    ```
    
5) Then run the pre-processing scripts in your terminal. This will need to run overnight:

    ```
    python3 -m eICU_preprocessing.run_all_preprocessing
    ```
    

   
## Running the models
1) Once you have run the pre-processing steps you can run all the models in your terminal. Set the working directory to the TPC-LoS-prediction, and run the following:

    ```
    python3 -m models.run_tpc
    ```
    
    Note that your experiment can be customised by using command line arguments e.g.
    
    ```
    python3 -m models.run_tpc --dataset eICU --task LoS --model_type tpc --n_layers 4 --kernel_size 3 --no_temp_kernels 10 --point_size 10 --last_linear_size 20 --diagnosis_size 20 --batch_size 64 --learning_rate 0.001 --main_dropout_rate 0.3 --temp_dropout_rate 0.1 
    ```
    
    Each experiment you run will create a directory within models/experiments. The naming of the directory is based on 
    the date and time that you ran the experiment (to ensure that there are no name clashes). The experiments are saved 
    in the standard trixi format: https://trixi.readthedocs.io/en/latest/_api/trixi.experiment.html.
    
2) The hyperparameter searches can be replicated by running:

    ```
    python3 -m models.hyperparameter_scripts.eICU.tpc
    ```
 
    Trixi provides a useful way to visualise effects of the hyperparameters (after running the following command, navigate to http://localhost:8080 in your browser):
    
    ```
    python3 -m trixi.browser --port 8080 models/experiments/hyperparameters/eICU/TPC
    ```
    
    The final experiments for the paper are found in models/final_experiment_scripts e.g.:
    
    ```
    python3 -m models.final_experiment_scripts.eICU.LoS.tpc
    ```
    
## References
Emma Rocheteau, Pietro Li `o, and Stephanie Hyland. 2021. Temporal pointwise convolutional networks for length of stay prediction in the intensive care unit. In Proceedings of the conference on health, inference, and learning, pages 58–68.

Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, and Aram Galstyan. Multitask Learning and Benchmarking with Clinical Time Series Data. Scientific Data, 6(96), 2019.


Shaojie Bai, J Zico Kolter, and Vladlen Koltun. 2018. An empirical evaluation of generic convolutional and
recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.

Seyedmostafa Sheikhalishahi, Vevake Balaraman, and Venet Osmani. 2019. Benchmarking machine learning models on eicu critical care dataset. arXiv preprint arXiv:1910.00964.
      
Huan Song, Deepta Rajan, Jayaraman Thiagarajan, and Andreas Spanias. 2018. Attend and diagnose: Clinical time series analysis using attention models. In Proceedings of the AAAI conference on artificial intelligence, volume 32.



