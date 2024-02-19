# Stable Diffusion을 통한 잠재공간 산책

<p align="center"><img src="https://github.com/ugiugi0823/pytorch-stable-diffusion-korean/assets/106899647/d10881c1-391e-4599-b2b5-b40b570e75e6" alt="image"></p>



⭐️거두 절미하고, 해보세요!⭐️
 ---> <a href="https://colab.research.google.com/drive/1Li66ZpHs_XLDQw9lcYD7WNmqpvRFqfon">
    <img src="https://colab.research.google.com/img/colab_favicon.ico" width="30" height="30" alt="Open In Colab"/>
</a>





---
### 텍스트 프롬프트 간 보간
<p align="center">
  <img src="./sd/src/P1.gif" alt="P1">
</p>

```
prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"
uncond_prompt = "worst quality"
```
---
### 수백 단계를 사용하여 훨씬 더 세밀한 보간
<p align="center">
  <img src="./sd/src/P2_resized.gif" alt="P1">
</p>

```
prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"
uncond_prompt = "worst quality"
```

---
### 네 가지 프롬프트 사이를 보간 (w / noise)
<p align="center">
  <img src="./sd/src/P3.jpg" alt="P3">
</p>

```
prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"
prompt_3 = "The eiffel tower in the style of starry night"
prompt_4 = "An architectural sketch of a skyscraper"
uncond_prompt = "worst quality"
```


---
### 네 가지 프롬프트 사이를 보간 (w / o noise)
<p align="center">
  <img src="./sd/src/P4.jpg" alt="P4">
</p>

```
prompt_1 = "A watercolor painting of a Golden Retriever at the beach"
prompt_2 = "A still life DSLR photo of a bowl of fruit"
prompt_3 = "The eiffel tower in the style of starry night"
prompt_4 = "An architectural sketch of a skyscraper"
uncond_prompt = "worst quality"
```
---
### Step_size걷기의 크기를 늘리거나 줄이도록 조정하여 직접 시도
<p align="center">
  <img src="./sd/src/P5_resized.gif" alt="P1">

```
walk_steps = 150
step_size = 0.005
prompt_5 = "The Eiffel Tower in the style of starry night"
uncond_prompt = "worst quality"

cond_context_5 = prepare_tokens(prompt_5,models, tokenizer, DEVICE)
uncond_context = prepare_tokens(uncond_prompt,models, tokenizer, DEVICE)
encoding = cond_context_5
delta = torch.ones_like(encoding) * step_size
```

---
###  단일 프롬프트에 대한 확산 노이즈 공간을 통한 원형 이동(Cos, Sin)
<p align="center">
  <img src="./sd/src/P6_resized.gif" alt="P1">

```
prompt_6 = "An oil paintings of cows in a field next to a windmill in Holland"
uncond_prompt = "worst quality"
```



## 가중치 및 토크나이저 파일 다운로드:

1. Download `vocab.json` and `merges.txt` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer and save them in the `data` folder
2. Download `v1-5-pruned-emaonly.ckpt` from https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main and save it in the `data` folder

## 테스트된 미세 조정 모델:

Just download the `ckpt` file from any fine-tuned SD (up to v1.5).

1. InkPunk Diffusion: https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main
2. Illustration Diffusion (Hollie Mengert): https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main

## Special thanks

Special thanks to the following repositories:

1. https://github.com/CompVis/stable-diffusion/
1. https://github.com/divamgupta/stable-diffusion-tensorflow
1. https://github.com/kjsman/stable-diffusion-pytorch
1. https://github.com/huggingface/diffusers/
2. https://keras.io/examples/generative/random_walks_with_stable_diffusion/

