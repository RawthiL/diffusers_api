import os
from typing import Any, List, Optional

from pydantic import BaseModel, field_validator


class DiffuserConfig(BaseModel):
    name: str
    xl_model: bool
    has_sag: bool
    cache_dir: str
    float_type: str
    use_safetensors: bool

    # Check for float type
    @field_validator("float_type")
    def valid_elements(cls, v):
        if v not in ["bf16", "fp16", "None"]:
            raise ValueError('Float type must be "fp16" or "bf16".')
        if v == "None":
            v = None
        return v

    # Check for cache dir existance
    @field_validator("cache_dir")
    def valid_elements(cls, v):
        if not os.path.exists(v):
            raise ValueError("Cache dir non existent found: %s" % v)
        return v


class Text2Img_Parameters(BaseModel):
    prompt: List[str]
    negative_prompt: List[str]
    height: int = 512
    width: int = 512
    samples: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = None
    safety_checker: bool = False
    self_attention: bool = None
    sag_scale: float = 1.0
    gen_output_type: str = "pil"
    output_type: str = "pil"
    return_dict: str = "true"


class ImageData(BaseModel):
    img_data: List[str]
    img_data_format: str = "raw_bytes"
    img_type: str = "RGB"
    img_height: int = 512
    img_width: int = 512


class Img2Img_Parameters(Text2Img_Parameters):
    base_img_data: Optional[ImageData] = None
    base_img: List[Any] = []
    aesthetic_score: float = 6
    negative_aesthetic_score: float = 2.5
    strength: float = 0.3


class Inpainting_Parameters(Img2Img_Parameters):
    mask_img_data: Optional[ImageData] = None
    mask_img: List[Any] = []


class Text2Img_Response(BaseModel):
    status: str = "fail"
    generationTime: float = 0.0
    id: int = 0
    output: List[str] = []
