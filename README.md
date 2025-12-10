# GTRM: Gated Recursive Reasoning with Tiny Networks

This is the codebase for the project: "GTRM: Gated Recursive Reasoning with Tiny Networks". GTRM is a recursive reasoning approach with gating that outperforms the original TRM baseline on Sudoku dataset, using a tiny 7M parameters neural network.

### Motivation

Recent advancements in Large Language Models, particularly the NeurIPS Best Paper [Gated Attention for Large Language Models](https://arxiv.org/pdf/2505.06708) (Qwen Team, 2025), have demonstrated the efficacy of gating mechanisms in regulating information flow and enhancing model expressivity. Drawing inspiration from this, we identify a potential limitation in standard TRM pipelines: the lack of explicit control over internal state updates often leads to the accumulation of redundant or noisy information.

To address this, we hypothesize that integrating a gating mechanism into the TRM pipeline can dynamically control the update process, thereby refining the representation learning. We explore this hypothesis by introducing three distinct gating strategies: Gated Context Injection, Gated Attention, and Gated Recurrence. Through comprehensive ablation studies, we systematically evaluate the impact of each gate on the model's performance, aiming to isolate the most effective mechanism for controlled state updating

### How GTRM works

<p align="center">
  <img src="https://raw.githubusercontent.com/KevinXu02/TRM/main/assets/GTRM.png" alt="GTRM"  style="width: 60%;">
</p>

We implemented three gating mechanisms to control the recursive update:

**1. Gated Context Injection (Input Stage)**
Controls the injection of static input context ($x$) into the latent state.

$$
h^{(t)} = z^{(t-1)} + \sigma(g^{(t)}) \cdot x
$$

**2. Gated Attention (Intermediate Stage)**
Modulates the features outputted by the attention mechanism.

$$
\tilde{z}^{(t)} = \text{Attn}(h^{(t)}) \odot \sigma(g^{(t)})
$$

**3. Gated Recurrent Update (Output Stage)**
Controls the trade-off between retaining old memory and accepting new updates (similar to GRU).    

$$
z^{(t)} = (1 - \sigma(g^{(t)})) \odot z^{(t-1)} + \sigma(g^{(t)}) \odot \tilde{z}^{(t)}
$$

> **Notation:**
> * $z^{(t)}$: Latent state at step $t$
> * $h^{(t)}$: Input to attention
> * $g^{(t)}$: Gate score
> * $\sigma$: Sigmoid activation
> * $\odot$: Element-wise multiplication

### Requirements

Installation should take a few minutes. For the smallest experiments on Sudoku-Extreme, you need 1 GPU with enough memory. With 5090, it takes around 7h to finish. Since different devices you have, install [requirements.txt](https://github.com/KevinXu02/TRM/blob/new/requirements.txt) is enough to prepare the env (Linux recommend).

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
# pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 examples, 8 augments
```

## Experiments

### Sudoku-Extreme:

```bash
python pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" epochs=50000 eval_interval=5000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=768 +run_name=exp5_gated_input ema=True
```

*Runtime:* < 8 hours

## Reference

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the TRM [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
