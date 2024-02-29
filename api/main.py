import os
import time

import data_utils
import diffuser_utils
import yaml
from basemodels import (
    DiffuserConfig,
    Img2Img_Parameters,
    Inpainting_Parameters,
    Text2Img_Parameters,
    Text2Img_Response,
)
from fastapi import FastAPI, HTTPException

###################################################
# LAUNCH
###################################################

# Read config
with open(os.getenv("CONFIG_FILE_PATH", "/config/config.yaml"), "r") as file:
    config_yaml = yaml.safe_load(file)

# Fill config
config = DiffuserConfig(
    name=config_yaml["model"]["name"],
    xl_model=config_yaml["model"]["is_xl"],
    has_sag=config_yaml["model"]["has_sag"],
    float_type=config_yaml["app"].get("float_type", "fp16"),
    use_safetensors=config_yaml["model"].get("use_safetensors", True),
    cache_dir=config_yaml["app"].get("hf_cache_dir", "/models"),
)
# Create diffuser
diffuser = diffuser_utils.Diffuser(config)

# Create serving app
app = FastAPI()

###################################################
# ENDPOINTS
###################################################

# -----------------------------------------------
# Text to Image Endpoint
# -----------------------------------------------
@app.post("/text2img")
def text2img_endpoint(input_request: Text2Img_Parameters) -> Text2Img_Response:

    tic = time.time()

    # Create response json
    response = Text2Img_Response()
    try:
        # Process the request
        images = diffuser.text2img(input_request)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred in the diffusion process.")

    try:
        # Encode the image into a string
        response.output = data_utils.images_to_data(input_request.output_type, images)

        # Set success and time
        response.id = 0
        response.status = "success"
        response.generationTime = time.time() - tic

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred while encoding the resulting image/s.")

    return response


# -----------------------------------------------
# Image+Text to Image Endpoint
# -----------------------------------------------
@app.post("/img2img")
def text2img_endpoint(input_request: Img2Img_Parameters) -> Text2Img_Response:

    tic = time.time()
    # Create response json
    response = Text2Img_Response()

    try:
        input_request.base_img = data_utils.data_to_images(input_request.base_img_data)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Unable to extract base image.")

    try:
        # Process the request
        images = diffuser.img2img(input_request)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred in the diffusion process.")

    try:
        # Encode the image into a string
        response.output = data_utils.images_to_data(input_request.output_type, images)

        # Set success and time
        response.id = 0
        response.status = "success"
        response.generationTime = time.time() - tic

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred while encoding the resulting image/s.")

    return response


# -----------------------------------------------
# Image+Text to Image Endpoint
# -----------------------------------------------
@app.post("/inpainting")
def inpainting_endpoint(input_request: Inpainting_Parameters) -> Text2Img_Response:

    tic = time.time()
    # Create response json
    response = Text2Img_Response()

    try:
        input_request.base_img = data_utils.data_to_images(input_request.base_img_data)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Unable to extract base image.")

    try:
        input_request.mask_img = data_utils.data_to_images(input_request.mask_img_data)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail="Unable to extract mask image.")

    try:
        # Process the request
        images = diffuser.inpainting(input_request)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred in the diffusion process.")

    try:
        # Encode the image into a string
        response.output = data_utils.images_to_data(input_request.output_type, images)

        # Set success and time
        response.id = 0
        response.status = "success"
        response.generationTime = time.time() - tic

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred while encoding the resulting image/s.")

    return response
