# Deployment


To deploy a new model into integration:
1. Update the corresponding inference config file in the `atlantes/atlas/config` directory. \
2. Update the model path in the `pipeline.py` file.
3. Merge into develop (in flux if this will trigger deployment)
