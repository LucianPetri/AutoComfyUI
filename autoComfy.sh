#!/bin/bash

# Prompt the user for the CIVITAI API key
read -r -p "Enter your CIVITAI API key: " CIVITAI_API_KEY

# Define the API key
# CIVITAI_API_KEY="example_key"
TOKEN="token=$CIVITAI_API_KEY"

# Define the Git URLs
GIT_COMFYUI="https://github.com/comfyanonymous/ComfyUI"

# Define CIVITAI SD1.5 Models URLs
SD15_JUGGERNAUT_REBORN="https://civitai.com/api/download/models/274039"
SD15_DREAMSHAPER="https://civitai.com/api/download/models/128713"
SD15_REV_ANIMATED="https://civitai.com/api/download/models/428862"
SD15_GHOSTMIX="https://civitai.com/api/download/models/76907"
SD15_NEVERENDING_DREAM="https://civitai.com/api/download/models/64094"
SD15_CYBERREALISTIC="https://civitai.com/api/download/models/372799"
SD15_ABSOLUTEREALITY="https://civitai.com/api/download/models/132760"
SD15_REALISTIC_VISION="https://civitai.com/api/download/models/130072"
SD15_DELIBERATE_V6="https://huggingface.co/XpucT/Deliberate/resolve/main/Deliberate_v6.safetensors?download=true"

# Define CIVITAI SDXL Models URLs
SDXL_JUGGERNAUTXL="https://civitai.com/api/download/models/456194" # HiRes: 4xNMKD-Siax_200k with 15 Steps and 0.3 Denoise + 1.5 Upscale

# Define CIVITAI SDXL Turbo Models URLs
SDXL_TURBO_DREAMSHAPER="https://civitai.com/api/download/models/372799"

# Define CIVITAI SD1.5 LCM Models URLs
LCM_DREAMSHAPER="https://civitai.com/api/download/models/252914"
LCM_ABSOLUTEREALITY="https://civitai.com/api/download/models/256668"

# Define CIVITAI SD1.5 Hyper Models URLs
SD15_HYPER_REALISTIC_VISION="https://civitai.com/api/download/models/256668"

# Define Upscale Models URLs
UPSCALER_4X_NMKD_SUPERSCALE="https://civitai.com/api/download/models/156841"
UPSCALER_2X_REAL_ESRGAN="https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth?download=true"
UPSCALER_4X_REAL_ESRGAN="https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth?download=true"
UPSCALER_8X_REAL_ESRGAN="https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth?download=true"
UPSCALER_4X_ESRGAN="https://huggingface.co/utnah/esrgan/resolve/main/4xESRGAN.pth?download=true"

# Define AnimateDiff Models URLs
ANIMATEDIFF_V3_SD15_MM="https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt?download=true"
ANIMATEDIFF_MM_SD_v15="https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt?download=true"
ANIMATEDIFF_MM_SD_v14="https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt?download=true"
ANIMATEDIFF_ANIMATE_LCM="https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v.ckpt?download=true"
ANIMATEDIFF_ANIMATE_LCM_LORA="https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors?download=true"

# Define VAE Model URLs
VAE_ORANGEMIX="https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt"
VAE_CLEAR="https://civitai.com/api/download/models/88156"
VAE_BLESSED2="https://civitai.com/api/download/models/142467"
VAE_FT_MSE_840K="https://civitai.com/api/download/models/311162"
VAE_MASTER="https://civitai.com/api/download/models/218292"

# Define Embeddings Models URLs
EMBEDD_DEEP_NEGATIVE_32T="https://civitai.com/api/download/models/5281"
EMBEDD_FAST_NEGATIVE_V2="https://civitai.com/api/download/models/94057"
EMBEDD_BAD_DREAM="https://civitai.com/api/download/models/77169"
EMBEDD_UNREALISTIC_DREAM="https://civitai.com/api/download/models/77173"
EMBEDD_BAD_PROMPT="https://civitai.com/api/download/models/60095"
EMBEDD_JUGGERNAUT_NEGATIVE="https://civitai.com/api/download/models/86553"
EMBEDD_CYBERREALISTIC_NEGATIVE="https://civitai.com/api/download/models/82745"
EMBEDD_ADVENTURE_POSITIVE="https://civitai.com/api/download/models/8042"
EMBEDD_STYLE_SWIRL_POSITIVE="https://civitai.com/api/download/models/30926"
EMBEDD_DELIBERATE_NEGATIVE="https://civitai.com/api/download/models/36426"

# Define LoRa Models URLs
LORA_DETAIL_TWEAKER="https://civitai.com/api/download/models/62833"
LORA_MORE_DETAIL="https://civitai.com/api/download/models/87153"
LORA_EPI_NOISEOFFSET="https://civitai.com/api/download/models/16576"
LORA_GHIBLI_BG="https://civitai.com/api/download/models/125985"
LORA_GLOWING_RUNES="https://civitai.com/api/download/models/93640"
LORA_VECTOR_ILLUSTRATION="https://civitai.com/api/download/models/198960"
LORA_INK_SCENERY="https://civitai.com/api/download/models/83390"
LORA_AURORAL="https://civitai.com/api/download/models/156828"
LORA_SCIFI_ENVIRONMENT="https://civitai.com/api/download/models/113765"
LORA_PASEER_MYTHOLOGY="https://civitai.com/api/download/models/104866"
LORA_RAL_BONES="https://civitai.com/api/download/models/302510"
LORA_RAL_CRYSTALS="https://civitai.com/api/download/models/238435"
LORA_RAL_LIGHTBULBS="https://civitai.com/api/download/models/322772"
LORA_RAL_GLITCH="https://civitai.com/api/download/models/322748"
LORA_RAL_FRACTAL_GEOMETRY="https://civitai.com/api/download/models/314363"
LORA_RAL_LAVA="https://civitai.com/api/download/models/265372"
LORA_RAL_LIQUID_FLOW="https://civitai.com/api/download/models/259228"
LORA_RAL_OPAL="https://civitai.com/api/download/models/303997"
LORA_RAL_DISSOLVE="https://civitai.com/api/download/models/314246"
LORA_RAL_MOLD="https://civitai.com/api/download/models/314302"
LORA_RAL_PENROSE_GEOMETRY="https://civitai.com/api/download/models/314218"
LORA_RAL_3D_WAVE="https://civitai.com/api/download/models/313991"
LORA_RAL_MELTING="https://civitai.com/api/download/models/313912"
LORA_RAL_BISMUTH="https://civitai.com/api/download/models/304077"
LORA_RAL_ELECTRICITY="https://civitai.com/api/download/models/301404"
LORA_RAL_MANDALA_COLOR_SWIRL="https://civitai.com/api/download/models/297350"
LORA_RAL_BLUE_RESIN="https://civitai.com/api/download/models/296564"
LORA_RAL_CHROME="https://civitai.com/api/download/models/276570"
LORA_RAL_FEATHER_CLOTEHS="https://civitai.com/api/download/models/275242"
LORA_RAL_AMBER="https://civitai.com/api/download/models/275216"
LORA_RAL_ORIGAMI="https://civitai.com/api/download/models/266928"
LORA_RAL_POLYGON="https://civitai.com/api/download/models/264506"
LORA_RAL_OVERGROWN="https://civitai.com/api/download/models/264449"
LORA_RAL_FIRE_ICE="https://civitai.com/api/download/models/261227"
LORA_RAL_CHOCOLATE="https://civitai.com/api/download/models/259150"
LORA_RAL_HORROR_SKELETON="https://civitai.com/api/download/models/239698"
LORA_RAL_ALIEN="https://civitai.com/api/download/models/239658"
LORA_RAL_MUSHROOMS="https://civitai.com/api/download/models/238514"
LORA_RAL_ACID_SLIME="https://civitai.com/api/download/models/238339"
LORA_RAL_GLOWING_SKULLS="https://civitai.com/api/download/models/47106"
LORA_RAL_COPPER_WIRE="https://civitai.com/api/download/models/314137"



########## Cloning ComfyUI repository

echo "Cloning ComfyUI repository..."
git clone $GIT_COMFYUI
echo "ComfyUI repository cloned!"

echo "Changing directory to ComfyUI..."
cd ComfyUI

########## Installing pytorch
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
echo "PyTorch installed!"
########## Installing Dependecies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed!"

# # Prompt the user to choose whether to download all models
# read -r -p "Do you want to download all models? ((a)all/(n)none/(s)select): " DOWNLOAD_MODELS

LORA_MODELS_FOLDER="models/loras"
CONTROLNET_MODELS_FOLDER="models/controlnet"
EMBEDDINGS_MODELS_FOLDER="models/embeddings"
VAE_MODELS_FOLDER="models/vae"
UPSCALE_MODELS_FOLDER="models/upscale"
ANIMATEDIFF_MODELS_FOLDER="models/animatediff"
SD15_MODELS_FOLDER="models/checkpoints/sd15"
SDXL_MODELS_FOLDER="models/checkpoints/sdxl"
SDXL_TURBO_MODELS_FOLDER="models/checkpoints/sdxl_turbo"
LCM_MODELS_FOLDER="models/checkpoints/lcm"
HYPER_MODELS_FOLDER="models/checkpoints/hyper"

# Download all models
echo "Downloading all models..."

# creating directories
mkdir -p $LORA_MODELS_FOLDER
mkdir -p $CONTROLNET_MODELS_FOLDER
mkdir -p $EMBEDDINGS_MODELS_FOLDER
mkdir -p $VAE_MODELS_FOLDER
mkdir -p $UPSCALE_MODELS_FOLDER
mkdir -p $ANIMATEDIFF_MODELS_FOLDER
mkdir -p $SD15_MODELS_FOLDER
mkdir -p $SDXL_MODELS_FOLDER
mkdir -p $SDXL_TURBO_MODELS_FOLDER
mkdir -p $LCM_MODELS_FOLDER
mkdir -p $HYPER_MODELS_FOLDER


# Downloading SD1.5 Models
echo "Downloading SD1.5 Models..."
# wget -O $SD15_MODELS_FOLDER/SD15_JUGGERNAUT_REBORN.safetensors $SD15_JUGGERNAUT_REBORN?$TOKEN
wget -O $SD15_MODELS_FOLDER/SD15_DREAMSHAPER.safetensors $SD15_DREAMSHAPER?$TOKEN
wget -O $SD15_MODELS_FOLDER/SD15_REV_ANIMATED.safetensors $SD15_REV_ANIMATED?$TOKEN
# wget -O $SD15_MODELS_FOLDER/SD15_GHOSTMIX.safetensors $SD15_GHOSTMIX?$TOKEN
# wget -O $SD15_MODELS_FOLDER/SD15_NEVERENDING_DREAM.safetensors $SD15_NEVERENDING_DREAM?$TOKEN
wget -O $SD15_MODELS_FOLDER/SD15_CYBERREALISTIC.safetensors $SD15_CYBERREALISTIC?$TOKEN
# wget -O $SD15_MODELS_FOLDER/SD15_ABSOLUTEREALITY.safetensors $SD15_ABSOLUTEREALITY?$TOKEN
# wget -O $SD15_MODELS_FOLDER/SD15_REALISTIC_VISION.safetensors $SD15_REALISTIC_VISION?$TOKEN
# wget -O $SD15_MODELS_FOLDER/SD15_DELIBERATE_V6.safetensors $SD15_DELIBERATE_V6?$TOKEN

# Downloading SDXL Models
echo "Downloading SDXL Models..."
# wget -O $SDXL_MODELS_FOLDER/SDXL_JUGGERNAUTXL.safetensors $SDXL_JUGGERNAUTXL?$TOKEN

# Downloading SDXL Turbo Models
echo "Downloading SDXL Turbo Models..."
# wget -O $SDXL_TURBO_MODELS_FOLDER/SDXL_TURBO_DREAMSHAPER.safetensors $SDXL_TURBO_DREAMSHAPER?$TOKEN

# Downloading LCM Models
echo "Downloading LCM Models..."
wget -O $LCM_MODELS_FOLDER/LCM_DREAMSHAPER.safetensors "$LCM_DREAMSHAPER?$TOKEN"
wget -O $LCM_MODELS_FOLDER/LCM_ABSOLUTEREALITY.safetensors "$LCM_ABSOLUTEREALITY?$TOKEN"

# Downloading Upscale Models
echo "Downloading Upscale Models..."
wget -O $UPSCALE_MODELS_FOLDER/4X_NMKD_SUPERSCALE.safetensors $UPSCALER_4X_NMKD_SUPERSCALE?$TOKEN
wget -O $UPSCALE_MODELS_FOLDER/2X_REAL_ESRGAN.pth $UPSCALER_2X_REAL_ESRGAN?$TOKEN
wget -O $UPSCALE_MODELS_FOLDER/4X_REAL_ESRGAN.pth $UPSCALER_4X_REAL_ESRGAN?$TOKEN
wget -O $UPSCALE_MODELS_FOLDER/8X_REAL_ESRGAN.pth $UPSCALER_8X_REAL_ESRGAN?$TOKEN
wget -O $UPSCALE_MODELS_FOLDER/4X_ESRGAN.pth $UPSCALER_4X_ESRGAN?$TOKEN

# Downloading AnimateDiff Models
echo "Downloading AnimateDiff Models..."

wget -O $ANIMATEDIFF_MODELS_FOLDER/ANIMATEDIFF_V3_SD15_MM.ckpt $ANIMATEDIFF_V3_SD15_MM?$TOKEN
wget -O $ANIMATEDIFF_MODELS_FOLDER/ANIMATEDIFF_MM_SD_v15.ckpt $ANIMATEDIFF_MM_SD_v15?$TOKEN
wget -O $ANIMATEDIFF_MODELS_FOLDER/ANIMATEDIFF_MM_SD_v14.ckpt $ANIMATEDIFF_MM_SD_v14?$TOKEN
wget -O $ANIMATEDIFF_MODELS_FOLDER/ANIMATEDIFF_ANIMATE_LCM.ckpt $ANIMATEDIFF_ANIMATE_LCM?$TOKEN
wget -O $ANIMATEDIFF_MODELS_FOLDER/ANIMATEDIFF_ANIMATE_LCM_LORA.safetensors $ANIMATEDIFF_ANIMATE_LCM_LORA?$TOKEN

# Downloading VAE Models
echo "Downloading VAE Models..."
wget -O $VAE_MODELS_FOLDER/VAE_ORANGEMIX.vae.pt $VAE_ORANGEMIX?$TOKEN
wget -O $VAE_MODELS_FOLDER/VAE_CLEAR.safetensors $VAE_CLEAR?$TOKEN
wget -O $VAE_MODELS_FOLDER/VAE_BLESSED2.safetensors $VAE_BLESSED2?$TOKEN
wget -O $VAE_MODELS_FOLDER/VAE_FT_MSE_840K.safetensors $VAE_FT_MSE_840K?$TOKEN
wget -O $VAE_MODELS_FOLDER/VAE_MASTER.safetensors $VAE_MASTER?$TOKEN

# Downloading Embeddings Models
echo "Downloading Embeddings Models..."
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_DEEP_NEGATIVE_32T.safetensors $EMBEDD_DEEP_NEGATIVE_32T?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_FAST_NEGATIVE_V2.safetensors $EMBEDD_FAST_NEGATIVE_V2?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_BAD_DREAM.safetensors $EMBEDD_BAD_DREAM?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_UNREALISTIC_DREAM.safetensors $EMBEDD_UNREALISTIC_DREAM?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_BAD_PROMPT.safetensors $EMBEDD_BAD_PROMPT?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_JUGGERNAUT_NEGATIVE.safetensors $EMBEDD_JUGGERNAUT_NEGATIVE?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_CYBERREALISTIC_NEGATIVE.safetensors $EMBEDD_CYBERREALISTIC_NEGATIVE?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_ADVENTURE_POSITIVE.safetensors $EMBEDD_ADVENTURE_POSITIVE?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_STYLE_SWIRL_POSITIVE.safetensors $EMBEDD_STYLE_SWIRL_POSITIVE?$TOKEN
wget -O $EMBEDDINGS_MODELS_FOLDER/EMBEDD_DELIBERATE_NEGATIVE.safetensors $EMBEDD_DELIBERATE_NEGATIVE?$TOKEN

# Downloading LoRa Models
echo "Downloading LoRa Models..."
wget -O $LORA_MODELS_FOLDER/LORA_DETAIL_TWEAKER.safetensors $LORA_DETAIL_TWEAKER?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_MORE_DETAIL.safetensors $LORA_MORE_DETAIL?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_EPI_NOISEOFFSET.safetensors $LORA_EPI_NOISEOFFSET?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_GHIBLI_BG.safetensors $LORA_GHIBLI_BG?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_GLOWING_RUNES.safetensors $LORA_GLOWING_RUNES?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_VECTOR_ILLUSTRATION.safetensors $LORA_VECTOR_ILLUSTRATION?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_INK_SCENERY.safetensors $LORA_INK_SCENERY?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_AURORAL.safetensors $LORA_AURORAL?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_SCIFI_ENVIRONMENT.safetensors $LORA_SCIFI_ENVIRONMENT?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_PASEER_MYTHOLOGY.safetensors $LORA_PASEER_MYTHOLOGY?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_BONES.safetensors $LORA_RAL_BONES?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_CRYSTALS.safetensors $LORA_RAL_CRYSTALS?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_LIGHTBULBS.safetensors $LORA_RAL_LIGHTBULBS?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_GLITCH.safetensors $LORA_RAL_GLITCH?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_FRACTAL_GEOMETRY.safetensors $LORA_RAL_FRACTAL_GEOMETRY?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_LAVA.safetensors $LORA_RAL_LAVA?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_LIQUID_FLOW.safetensors $LORA_RAL_LIQUID_FLOW?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_OPAL.safetensors $LORA_RAL_OPAL?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_DISSOLVE.safetensors $LORA_RAL_DISSOLVE?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_MOLD.safetensors $LORA_RAL_MOLD?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_PENROSE_GEOMETRY.safetensors $LORA_RAL_PENROSE_GEOMETRY?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_3D_WAVE.safetensors $LORA_RAL_3D_WAVE?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_MELTING.safetensors $LORA_RAL_MELTING?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_BISMUTH.safetensors $LORA_RAL_BISMUTH?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_ELECTRICITY.safetensors $LORA_RAL_ELECTRICITY?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_MANDALA_COLOR_SWIRL.safetensors $LORA_RAL_MANDALA_COLOR_SWIRL?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_BLUE_RESIN.safetensors $LORA_RAL_BLUE_RESIN?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_CHROME.safetensors $LORA_RAL_CHROME?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_FEATHER_CLOTEHS.safetensors $LORA_RAL_FEATHER_CLOTEHS?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_AMBER.safetensors $LORA_RAL_AMBER?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_ORIGAMI.safetensors $LORA_RAL_ORIGAMI?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_POLYGON.safetensors $LORA_RAL_POLYGON?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_OVERGROWN.safetensors $LORA_RAL_OVERGROWN?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_FIRE_ICE.safetensors $LORA_RAL_FIRE_ICE?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_CHOCOLATE.safetensors $LORA_RAL_CHOCOLATE?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_HORROR_SKELETON.safetensors $LORA_RAL_HORROR_SKELETON?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_ALIEN.safetensors $LORA_RAL_ALIEN?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_MUSHROOMS.safetensors $LORA_RAL_MUSHROOMS?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_ACID_SLIME.safetensors $LORA_RAL_ACID_SLIME?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_GLOWING_SKULLS.safetensors $LORA_RAL_GLOWING_SKULLS?$TOKEN
wget -O $LORA_MODELS_FOLDER/LORA_RAL_COPPER_WIRE.safetensors $LORA_RAL_COPPER_WIRE?$TOKEN
wget -O $LORA_MODELS_FOLDER/ANIMATEDIFF_ANIMATE_LCM_LORA.safetensors $ANIMATEDIFF_ANIMATE_LCM_LORA?$TOKEN
echo "All models downloaded!"




# Process the user's choice

# if [[ $DOWNLOAD_MODELS == "all"|| $DOWNLOAD_MODELS == "a"||$DOWNLOAD_MODELS == "A"||$DOWNLOAD_MODELS == "ALL" ]]; then
#     # Download all models
#     echo "Downloading all models..."
#     # Add your code here to download all models

#     elif [[ $DOWNLOAD_MODELS == "n"||$DOWNLOAD_MODELS == "N"||$DOWNLOAD_MODELS == "no"||$DOWNLOAD_MODELS == "NO" ]]; then
#     # Do not download any models
#     echo "Not downloading any models."
# else
#     # Define the list of models
#     models=("SD15_JUGGERNAUT_REBORN" "SD15_DREAMSHAPER" "SD15_REV_ANIMATED" "SD15_GHOSTMIX" "SD15_NEVERENDING_DREAM" "SD15_CYBERREALISTIC" "SD15_ABSOLUTEREALITY" "SD15_REALISTIC_VISION" "SD15_DELIBERATE_V6" "SDXL_JUGGERNAUTXL" "SDXL_TURBO_DREAMSHAPER" "LCM_DREAMSHAPER" "LCM_ABSOLUTEREALITY" "SD15_HYPER_REALISTIC_VISION" "QUIT")

#     # Prompt the user to select specific models to download
#     echo "Select models to download (separate by space):"

#     # Print the list of models
#     for i in "${!models[@]}"; do
#         printf "%s\t%s\n" "$i" "${models[$i]}"
#     done

#     # Read the user's selection
#     read -a selection

#     # Process the user's selection
#     for index in "${selection[@]}"; do
#         if [ "$index" -lt "${#models[@]}" -a "$index" -ge 0 ]; then
#             model=${models[$index]}
#             if [ "$model" == "QUIT" ]; then
#                 echo "Quitting..."
#                 break
#             else
#                 echo "Downloading $model..."
#                 # Add your code here to download the selected model
#                 case $model in
#                     "SD15_JUGGERNAUT_REBORN")
#                         wget -O SD15_JUGGERNAUT_REBORN.safetensors $SD15_JUGGERNAUT_RE
#             fi
#         else
#             echo "Invalid selection: $index. Please try again."
#         fi
#     done

# fi