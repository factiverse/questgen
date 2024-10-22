# QuestGen: Effectiveness of Question Generation Methods for Fact-Checking Applications


## Getting Started:
1. Clone this repository
1. Install requirements using `pip install -r requirements.txt`

## Adding a dataset
1. Add a folder to `data/` with the name of the new dataset (for this example, I will call the dataset `example`)
1. Include a `test.json` and `train.json` files with testing and training data. These should be located in the `data/example` directory.
1. The data must conform to the following format

    ```
    [
        {
            "input_text": "INPUT1",
            "target_text": "TARGET1",
            "prefix": "PREFIX1"
        },
        {
            "input_text": "INPUT2",
            "target_text": "TARGET2",
            "prefix": "PREFIX2"
        }
        ...
    ]
    ```
1. See the `data/sample` dataset for reference on how to add a dataset to this project.

## Creating Model Configs
1. Go to `src/model_configs`
1. Create a `.yaml` file, and name it appropriately
1. Fill out the following for your model. Make sure that everything in `*here*` is filled out approperiately

    ```
    data: data/*data-set-name*
    hyper parameters:
    evaluation_strategy: *epoch/step*
    fp16: *true/false*
    learning_rate: *lr*
    no_cuda: *true/false*
    num_train_epochs: *num*
    overwrite_output_dir: *true/false*
    per_device_eval_batch_size: *num*
    per_device_train_batch_size: *num*
    predict_with_generate: *true/false*
    weight_decay: *num(0-1)*
    metrics:
    bleu: *true/false*
    rouge: *true/false*
    model_checkpoint: #Include the model filepath here `models\...`
    output_dir: *models/*
    wandb_tags:
    - *tag 1*
    ```
1. Refer to `src/model_configs/config.yaml` for an example model configuration for the facebook/bart-base model

## Finetuning or Testing
1. To finetune or test, make sure you first have a model_config `.yaml` file. See above to creating a model config file.
1. To **finetune** a model given a model configuration file, run

    ```python src/Generate-Question-Finetune.py --config src/model_configs/*confile-file*```

1. To **test** a model given the model configuration file, run
    
    ```python src/Generate-Question-Test --config src/model_configs/*confile-file*```


<!-- ##Usage
Ensure that you are on a machine using CUDA.  -->
