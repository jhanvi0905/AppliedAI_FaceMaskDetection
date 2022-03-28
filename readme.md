## DESCRIPTION

Files 

1. dataProcess.py - module that derives and  processes the dataset converting them into batches- contains supporting functions for same\

2. train_model.py - contains the model class and training configurations deriving data from previous modules

3. eval.py - contains code that derives prediction on each image in testing folder and corresponding metrics

Directories:

1. Ouptut: Split data in terms of train, test and validation

2. output_models: Saved model configurations

3. Output_Predictions: Predictions for images in testing class 

4. Classified: Original Dataset


Training the Model:

python3 train_model.py -ep noOfEpochs -bs BatchSize -lr LearnRate
 
Set values in respective, -ep, -bs and -lr flags to set the training configuration.


Evaluating the Model and Generating the metrics:

 python3 eval.py
