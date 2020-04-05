"""Params for ADDA."""

# params for dataset and data loader
data_root = "data/xray"
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value,)
dataset_std = (dataset_std_value,)
batch_size = 32
image_size = 224

# params for source dataset
src_dataset = "chexpert"
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = "NIH"
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 100
# num of steps to output log
log_step_pre = 100 
# num of epochs to evaluate at
eval_step_pre = 5 
# num of epochs to save model
save_step_pre = 20
num_epochs = 200
# num of steps to output log
log_step = 100
# num of epochs to save model
save_step = 20
# num of epochs to evaluate at
eval_step = 10
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
