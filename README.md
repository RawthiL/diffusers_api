# Diffusers API

A simple API to run [Huggingface's diffusers](https://github.com/huggingface/diffusers) locally and expose them using a REST API.

### Testing

1. Build the docker image:
```bash
chmod +x ./build.sh
./build.sh
```

2. Edit the config file provided in `api/config`. You need to provide a models name or a path to where it will be mounted in the docker volume. See config file for details.

3. Run the docker image:
```bash
docker run \         
    --gpus '"device=0"' \
    -v $path_to_where_store_the_models:/models \
    -v $path_to_the_config_file_that_you_wish_to_use:/config/config.yaml \
    -p 80:80 \
    -it diffusers_api:latest 
```

4. See the provided notebook `API_test.ipynb` for examples on how to interact with the API.
