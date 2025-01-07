# Training

## Training QuickStart

All Ais models are trained in a similar fashion so we will use the end of sequence model as an example.


1. Configure atlantes/atlas/config/atlas_activity_real_time_config.yaml for the given experiment and optionally data_config.yaml
2. Ensure you have the appropriate environment variables set and credentials
3.  If on GCP run the training script:

```bash
python3  training_scripts/launch_train_atlas_activity_real_time.py
```
4. If on Beaker configure the beaker yaml file and launch the experiment:

```bash
beaker experiment create path/to/your/spec.yml
```


### Training models on beaker (optional/for AI2 employees/collaborators only)

Beaker is a collaborative platform for rapid, reproducible research developed by AI2. This section is only relevant to AI2 employees and collaborators with access to AI2's hardware. Note that running jobs on beaker is not needed/required to train ATLAS/Atlantes. Beaker is simply AI2's preferred method of managing experiemntal jobs on their own infrastructure.

To use beaker see the following [guide](https://beaker-docs.apps.allenai.org/) and https://beaker.org/ for more information.

To run a beaker experiment (after creating an account following the guide above), you will need to first build your image. This repo supports training either activity or entity based models, which are specified at build time via `--build-arg`. For example:


`docker build  -t <your-tag> .`


`beaker image create --name  <your-tag>  <your-tag>`

```bash
beaker experiment create path/to/your/spec.yml
```

Our experiments require GCP authentication to stream data from a GCP bucket to the machine that is running the beaker. Authentication is enabled by a beaker secret (see ./beaker-config.yml), associated with a service account: ais-beaker@skylight-proto-1.iam.gserviceaccount.com that was read priveleges to the required bucket.

`gcloud iam service-accounts keys create credentials.json --iam-account ais-beaker@skylight-proto-1.iam.gserviceaccount.com`

The, create a beaker secret https://beaker-docs.apps.allenai.org/concept/secrets.html

`cat credentials.json | beaker secret write GCP_CREDENTIALS`

`beaker secret write WANDB_API_KEY {your wandb api key}`
`beaker secret write GCP_CREDENTIALS_PATH /etc/credentials/credentials.json`

Note that you should be mindful about egress costs for that training experiment, because egress for ~10 TB may be as much as $1000/epoch in egress fees alone (i.e. not including the cost of GPU compute). Special note: to SSH into the beaker machines (such as the cluster on which your experiment is running), you must add your public key [here](https://bridge.allenai.org/ssh). So, when training large experiments on Jupiter cluster configure your experiments to use WEKA storage.


## Updating Configurations

There are two main ways to update experimental configurations for beaker experiments. Initially, there will always be a config baked into the image that sets different variables within the experiment.

1. Override Config in you beaker experiment spec. \
  To make small changes to the config add additional args to the command in the beaker spec. For example:
  `python -u src/atlantes/training_scripts/launch_train_atlas_activity_real_time.py NUM_GPUS=8 MODEL_SAVE_DIR=/models EVAL_BEFORE_TRAIN=False MODEL_LOAD_DIR=src/atlantes/models DROPOUT_P=0.3 NUM_DATA_WORKERS=16 TRAIN_BATCH_SIZE=256 WEIGHT_DECAY=0.01 VAL_BATCH_SIZE=256 N_EPOCHS=30 LEARNING_RATE=0.0001 `

  This is ideal for changing vars with simple types like int and str.

2. Update the config locally and launch with launch_beaker_experiment.py. \
  This is ideal for changing vars with more complex types like dicts.
  Or when the experimental configs are changing drastically so you don't need to remember the default values baked into the image
  Be sure to override the Default variables like workspace, budget, etc as need with the cli

For further beaker questions, see the [beaker docs](https://beaker-docs.apps.allenai.org/)



### Training on GCP VMs

We use GCP Deep Learning VMs for GPU acceleration. We recommend ubuntu family of torch base images.

```bash
export IMAGE_FAMILY="pytorch-2-1-cu121-ubuntu-2004-py310"
export ZONE="us-west1-b"
export INSTANCE_NAME="pb-atlas-pretrain-4"

gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --accelerator="type=nvidia-tesla-v100,count=1" \
  --metadata="install-nvidia-driver=True" \
  --machine-type="n1-highmem-8"
```







## Training file Reference
