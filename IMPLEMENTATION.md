# Gemma 4 31B LoRA Finetune for GBM MRI VQA — Implementation Spec

## 目标

在 GBM MRI 数据上 LoRA 微调 Gemma 4 31B-IT，输入为单序列 MRI volume 抽取的多张 axial slice，输出为对应序列的放射学描述。

## 硬件

- 8× NVIDIA H200 (141GB each, 1128GB total), Hopper SM 9.0
- 31B bf16 权重 (~62GB) 单卡即可放下 → ZeRO-2 足够,无需 ZeRO-3/FSDP
  param sharding
- Hopper 支持 FP8(FP4 仍 Blackwell-only);本实现用 bf16 + LoRA,
  attn 默认 `sdpa`(host 装好 flash-attn 后可换 `flash_attention_2`)
- (原 spec 假设 4×A6000 48GB,该前提下 ZeRO-2 会 OOM,需 ZeRO-3/QLoRA;
  现按 8×H200 调参,见 configs/train_config.yaml)

## 技术栈

- PyTorch 2.4+
- `transformers >= 4.49`（支持 Gemma 4）
- `peft` for LoRA
- `accelerate` for multi-GPU
- `bitsandbytes` for QLoRA（可选）
- `deepspeed` ZeRO-2（如果 OOM 用 ZeRO-3）
- `nibabel` for NIfTI
- `wandb` for logging

## 数据格式

输入 JSONL，每行一个 sample:

```json
{
  "case_id": "case_001",
  "sequence": "t1c",
  "nifti_path": "/path/to/case_001/t1c.nii.gz",
  "mask_path": "/path/to/case_001/tumor_mask.nii.gz",
  "report": "Right frontal lobe shows a 3.2 cm enhancing mass..."
}
```

## 模块分解

- **Module 1** `data/slice_extractor.py` — extract/normalize N axial slices.
- **Module 2** `data/dataset.py` — JSONL -> processor inputs, with disk cache.
- **Module 3** `data/prompt_builder.py` — Gemma multi-image chat template.
- **Module 4** `model/load_model.py` — load model + attach LoRA.
- **Module 5** `training/train.py` — Trainer + custom collator + label mask.
- **Module 6** `configs/ds_zero2.json` — DeepSpeed ZeRO-2 (ZeRO-3 fallback).
- **Module 7** `eval/inference.py` — load checkpoint, generate captions.
- **Module 8** `scripts/preprocess.py` — offline slice caching.

## 项目结构 / 实现顺序

slice_extractor -> prompt_builder -> load_model -> dataset+collator ->
preprocess -> train (overfit 10 cases first) -> full train -> inference.

## 给 Claude Code 的实现约束

1. 每个 module 实现完先写 minimal test，跑通再继续。
2. 不发明 transformers/peft API。
3. Gemma 4 processor 用法与 Gemma 3 不同（2D RoPE、可变 token budget）。
4. 关键参数放 YAML config，不硬编码。
5. 路径用 `pathlib.Path`。
6. 训练 loop 前 1-batch dry run。
7. 用 logging 打印 loss / grad norm / lr。
8. ImportError 先 `pip install`。

## 常见踩坑

1. Gemma 4 module 命名与 Gemma 3 不同 — LoRA target 先 print 确认。
2. `remove_unused_columns=False` 必须开，否则 `pixel_values` 被 drop。
3. Label masking 要正确处理 multi-image chat template token 边界。
4. DeepSpeed + LoRA + gradient checkpointing 一起用，显式
   `gradient_checkpointing_kwargs={"use_reentrant": False}`。
5. vLLM 部署 LoRA：`--enable-lora --lora-modules`。

## 实现说明 (this repo)

环境无 GPU / 无 transformers，模型加载与训练无法在此 runtime 验证。
`Gemma4ForConditionalGeneration` / `Gemma4Processor` 通过 `Auto*` +
可配置 `model_id` 加载，以适配安装的 `transformers` 版本所提供的具体类名；
需在训练机上验证的位置已在代码中标注。
