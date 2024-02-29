import torch
from basemodels import (
    DiffuserConfig,
    Img2Img_Parameters,
    Inpainting_Parameters,
    Text2Img_Parameters,
)
from diffusers import (
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionSAGPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)


class Diffuser:
    def __init__(
        self,
        config: DiffuserConfig,
    ):
        super().__init__()

        # Set model type
        self.xl_model = config.xl_model

        # Base pipeline, load unet, vae, etc.
        if self.xl_model:
            # Text to Image
            self.TXT2IM_pipeline = StableDiffusionXLPipeline.from_pretrained(
                config.name,
                torch_dtype=torch.float16,
                use_safetensors=config.use_safetensors,
                variant=config.float_type,
                cache_dir=config.cache_dir,
            )
            # Recycle networks for consecute pipelines
            # Impainting pipeline
            self.IMP_pipeline = StableDiffusionXLInpaintPipeline(
                vae=self.TXT2IM_pipeline.vae,
                text_encoder=self.TXT2IM_pipeline.text_encoder,
                text_encoder_2=self.TXT2IM_pipeline.text_encoder_2,
                tokenizer=self.TXT2IM_pipeline.tokenizer,
                tokenizer_2=self.TXT2IM_pipeline.tokenizer_2,
                unet=self.TXT2IM_pipeline.unet,
                scheduler=self.TXT2IM_pipeline.scheduler,
                image_encoder=self.TXT2IM_pipeline.image_encoder,
                feature_extractor=self.TXT2IM_pipeline.feature_extractor,
                force_zeros_for_empty_prompt=self.TXT2IM_pipeline.config.force_zeros_for_empty_prompt,
            )
            # Image+Text to Image pipeline
            self.IM2IM_pipeline = StableDiffusionXLImg2ImgPipeline(
                vae=self.TXT2IM_pipeline.vae,
                text_encoder=self.TXT2IM_pipeline.text_encoder,
                text_encoder_2=self.TXT2IM_pipeline.text_encoder_2,
                tokenizer=self.TXT2IM_pipeline.tokenizer,
                tokenizer_2=self.TXT2IM_pipeline.tokenizer_2,
                unet=self.TXT2IM_pipeline.unet,
                scheduler=self.TXT2IM_pipeline.scheduler,
                image_encoder=self.TXT2IM_pipeline.image_encoder,
                feature_extractor=self.TXT2IM_pipeline.feature_extractor,
                force_zeros_for_empty_prompt=self.TXT2IM_pipeline.config.force_zeros_for_empty_prompt,
            )
        else:
            # Text to Image
            if config.has_sag:
                self.TXT2IM_pipeline = StableDiffusionSAGPipeline.from_pretrained(
                    config.name,
                    torch_dtype=torch.float16,
                    use_safetensors=config.use_safetensors,
                    variant=config.float_type,
                    cache_dir=config.cache_dir,
                )
            else:
                self.TXT2IM_pipeline = DiffusionPipeline.from_pretrained(
                    config.name,
                    torch_dtype=torch.float16,
                    use_safetensors=config.use_safetensors,
                    variant=config.float_type,
                    cache_dir=config.cache_dir,
                )

            # Recycle networks for consecute pipelines
            self.IM2IM_pipeline = StableDiffusionImg2ImgPipeline(
                vae=self.TXT2IM_pipeline.vae,
                text_encoder=self.TXT2IM_pipeline.text_encoder,
                tokenizer=self.TXT2IM_pipeline.tokenizer,
                unet=self.TXT2IM_pipeline.unet,
                scheduler=self.TXT2IM_pipeline.scheduler,
                image_encoder=self.TXT2IM_pipeline.image_encoder,
                feature_extractor=self.TXT2IM_pipeline.feature_extractor,
                safety_checker=self.TXT2IM_pipeline.safety_checker,
            )

            # IMP_pipeline = AutoPipelineForInpainting.from_config(pipeline)
            self.IMP_pipeline = StableDiffusionInpaintPipeline(
                vae=self.TXT2IM_pipeline.vae,
                text_encoder=self.TXT2IM_pipeline.text_encoder,
                tokenizer=self.TXT2IM_pipeline.tokenizer,
                unet=self.TXT2IM_pipeline.unet,
                scheduler=self.TXT2IM_pipeline.scheduler,
                image_encoder=self.TXT2IM_pipeline.image_encoder,
                feature_extractor=self.TXT2IM_pipeline.feature_extractor,
                safety_checker=self.TXT2IM_pipeline.safety_checker,
            )

        # Send to GPU, all pipelines share the same model in GPU
        device = torch.device("cuda:0")
        self.TXT2IM_pipeline = self.TXT2IM_pipeline.to(device)
        self.IMP_pipeline = self.IMP_pipeline.to(device)
        self.IM2IM_pipeline = self.IM2IM_pipeline.to(device)

    def get_generator(self, params: Text2Img_Parameters):
        if params.seed != None:
            g_cpu = torch.Generator()
            generator = g_cpu.manual_seed(params.seed)
        else:
            generator = None
        return generator

    def text2img(self, params: Text2Img_Parameters):
        print("Processing text2img...")
        result = self.TXT2IM_pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=self.get_generator(params),
            output_type=params.gen_output_type,
            return_dict="true",
            sag_scale=params.sag_scale,
        )

        return result.images

    def img2img(self, params: Img2Img_Parameters):
        print("Processing img2img...")
        result = self.IM2IM_pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            image=params.base_img,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=self.get_generator(params),
            output_type=params.gen_output_type,
            return_dict="true",
            aesthetic_score=params.aesthetic_score,
            negative_aesthetic_score=params.negative_aesthetic_score,
            strength=params.strength,
        )

        return result.images

    def inpainting(self, params: Inpainting_Parameters):
        print("Processing inpainting...")
        result = self.IMP_pipeline(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            image=params.base_img,
            mask_image=params.mask_img,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=self.get_generator(params),
            output_type=params.gen_output_type,
            return_dict="true",
            aesthetic_score=params.aesthetic_score,
            negative_aesthetic_score=params.negative_aesthetic_score,
            strength=params.strength,
        )

        return result.images
