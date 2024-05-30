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

# Prompt the user to choose whether to download all models
read -r -p "Do you want to download all models? ((a)all/(n)no/(s)select): " DOWNLOAD_MODELS

if [[ $DOWNLOAD_MODELS == "all"|| $DOWNLOAD_MODELS == "a"||$DOWNLOAD_MODELS == "A"||$DOWNLOAD_MODELS == "ALL" ]]; then
    # Download all models
    echo "Downloading all models..."
    # Add your code here to download all models
    elif [[ $DOWNLOAD_MODELS == "n"||$DOWNLOAD_MODELS == "N"||$DOWNLOAD_MODELS == "no"||$DOWNLOAD_MODELS == "NO" ]]; then
    # Do not download any models
    echo "Not downloading any models."
else
    # Define the list of models
    models=("SD15_JUGGERNAUT_REBORN" "SD15_DREAMSHAPER" "SD15_REV_ANIMATED" "SD15_GHOSTMIX" "SD15_NEVERENDING_DREAM" "SD15_CYBERREALISTIC" "SD15_ABSOLUTEREALITY" "SD15_REALISTIC_VISION" "SD15_DELIBERATE_V6" "SDXL_JUGGERNAUTXL" "SDXL_TURBO_DREAMSHAPER" "LCM_DREAMSHAPER" "LCM_ABSOLUTEREALITY" "SD15_HYPER_REALISTIC_VISION" "QUIT")
    
    # Prompt the user to select specific models to download
    echo "Select models to download (separate by space):"
    
    # Print the list of models
    for i in "${!models[@]}"; do
        printf "%s\t%s\n" "$i" "${models[$i]}"
    done
    
    # Read the user's selection
    read -a selection
    
    # Process the user's selection
    for index in "${selection[@]}"; do
        if [ "$index" -lt "${#models[@]}" -a "$index" -ge 0 ]; then
            model=${models[$index]}
            if [ "$model" == "QUIT" ]; then
                echo "Quitting..."
                break
            else
                echo "Downloading $model..."
                # Add your code here to download the selected model
            fi
        else
            echo "Invalid selection: $index. Please try again."
        fi
    done
    
fi