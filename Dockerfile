# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
RUN git clone https://github.com/city96/ComfyUI-GGUF custom_nodes/ComfyUI-GGUF

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base as downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Download checkpoints/vae/LoRA to include in image based on model type

RUN wget --header="Authorization: Bearer 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" -O models/diffusion_models/sd3.5_large.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors
RUN wget --header="Authorization: Bearer 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" -O models/diffusion_models/sd3.5_large-turbox.safetensors https://huggingface.co/tensorart/stable-diffusion-3.5-large-TurboX/resolve/main/TensorArt-SD3.5-Large-TurboX.safetensors
RUN wget --header="Authorization: Bearer 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" -O models/vae/sdvae.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/vae/diffusion_pytorch_model.safetensors
RUN wget --header="Authorization: Bearer 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" -O models/controlnet/sd3.5_large_controlnet_blur.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/sd3.5_large_controlnet_blur.safetensors
RUN wget --header="Authorization: Bearer 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" -O models/controlnet/sd3.5_large_controlnet_canny.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/sd3.5_large_controlnet_canny.safetensors
RUN wget --header="Authorization: Bearer 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" -O models/controlnet/sd3.5_large_controlnet_depth.safetensors https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets/resolve/main/sd3.5_large_controlnet_depth.safetensors
RUN wget -O models/text_encoders/clip_l.safetensors https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_l.safetensors
RUN wget -O models/text_encoders/clip_g.safetensors https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/clip_g.safetensors
RUN wget -O models/text_encoders/t5xxl_fp16.safetensors https://huggingface.co/Comfy-Org/stable-diffusion-3.5-fp8/resolve/main/text_encoders/t5xxl_fp16.safetensors



# Stage 3: Final image
FROM base as final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start the container
CMD /start.sh
