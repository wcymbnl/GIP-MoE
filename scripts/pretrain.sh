#!/bin/bash
set -e

if [ -n "${CONDA_ENV_BIN:-}" ]; then
    export PATH="${CONDA_ENV_BIN}:$PATH"
fi

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:${LD_PRELOAD}
fi
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/llmdet_torch_extensions}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/llmdet_mplconfig}
export HF_HOME=${HF_HOME:-/tmp/llmdet_hf_home}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/llmdet_triton}
export TRANSFORMERS_OFFLINE=1

DEEPSPEED_BIN=${DEEPSPEED_BIN:-deepspeed}
DATA_PATH=${DATA_PATH:-/home/ldl/ldl/fourth/LLMDet/grounding_data/llava_cap/LLaVA-ReCap-558K_tag.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-/home/ldl/ldl/fourth/LLMDet/grounding_data/llava_cap/images}
VISION_TOWER_WEIGHT=${VISION_TOWER_WEIGHT:-/home/ldl/ldl/fourth/LLMDet/huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth}
BERT_BASE_PATH=${BERT_BASE_PATH:-/home/ldl/ldl/fourth/LLMDet/huggingface/bert-base-uncased/}
STAGE1_MODE=${STAGE1_MODE:-full}

case "${STAGE1_MODE}" in
    full)
        MODEL_PATH=${MODEL_PATH:-/home/ldl/ldl/fourth/LLMDet/huggingface/llava-onevision-qwen2-0.5b-ov-2/}
        VISION_TOWER=${VISION_TOWER:-grounding_dino_mixed}
        PRETRAIN_MM_MLP_ADAPTER=${PRETRAIN_MM_MLP_ADAPTER:-}
        OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/llava-grounding_dino_mixed-pretrain-full}
        NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
        MAX_STEPS=${MAX_STEPS:--1}
        SAVE_STEPS=${SAVE_STEPS:-1000}
        SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}
        LEARNING_RATE=${LEARNING_RATE:-1e-3}
        ;;
    *)
        echo "Unknown STAGE1_MODE=${STAGE1_MODE}. Use: author_tiny, author_fast, or full." >&2
        exit 1
        ;;
esac

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}

EXTRA_PROJECTOR_ARGS=()
if [ -n "${PRETRAIN_MM_MLP_ADAPTER}" ]; then
    EXTRA_PROJECTOR_ARGS+=(--pretrain_mm_mlp_adapter "${PRETRAIN_MM_MLP_ADAPTER}")
fi

echo "Stage 1 mode: ${STAGE1_MODE}"
echo "Model path: ${MODEL_PATH}"
echo "Vision tower: ${VISION_TOWER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max steps: ${MAX_STEPS}"
echo "Learning rate: ${LEARNING_RATE}"

WANDB__SERVICE_WAIT=300 "${DEEPSPEED_BIN}" train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "${MODEL_PATH}" \
    --version qwen_1_5 \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --vision_tower_weight_path "${VISION_TOWER_WEIGHT}" \
    --mm_projector_type mlp2x_gelu \
    "${EXTRA_PROJECTOR_ARGS[@]}" \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --load_ram False \
    --grounding_dino_config configs/gip_moe_swin_t_for_alignment.py \
    --bert_base_path "${BERT_BASE_PATH}" \
    --bf16 False \
    --fp16 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --lazy_preprocess True \
    --report_to none
