# sNeuron-TST (hate-speech TST fork)

My working copy of the official [sNeuron-TST](https://github.com/wenlai-lavine/sNeuron-TST)
code — *Style-Specific Neurons for Steering LLMs in Text Style Transfer*
([Lai et al., EMNLP 2024](http://arxiv.org/abs/2410.00593)). I used it as a
**baseline** for our paper *[LLM in the Loop: Creating the ParaDeHate Dataset for
Hate Speech Detoxification](https://arxiv.org/abs/2506.01484)*, where we needed to
compare against existing style-transfer methods on hate-speech detoxification
(rewriting toxic text into a neutral style while preserving meaning). That meant
running the neuron-identification + DoLa decoding pipeline on our hate-speech data.

Getting it to run on a current environment took a few fixes, which are the point
of this repo.

## What I changed

- **Fixed a broken import that stops the repo from loading.** `Our/dola.py`
  imported `LLamaQaStoppingCriteria` from `transformers.generation.stopping_criteria`,
  which isn't part of the public `transformers` API and raises `ImportError` on
  current releases. I define the class locally instead (same `\nQ:` stop
  behaviour). Submitted upstream: [wenlai-lavine/sNeuron-TST#6](https://github.com/wenlai-lavine/sNeuron-TST/pull/6).
- **Python 3.11 compatibility** — worked around the `fake_tensor` path that
  breaks under 3.11.
- **`np.log` → `torch.log`** in the relative-top filter, to keep everything on
  one device/dtype.
- **Added a Hate_Speech pipeline** — dataset split (`data_pre/Hate_Speech`),
  activation storage, neuron identification, and a select-mode run so the method
  can be evaluated on hate-speech data.

## Auth

The dataset/model loaders read a Hugging Face token from the environment rather
than hardcoding one:

```bash
export HF_TOKEN=hf_your_token_here
```

## Layout

Files live under `src/` (the pipeline stages from the original README: `activation.py`,
`identify.py`, `Analysis/select_neurons.py`, `Our/run_gen_dola.py`). See
[`src/README.md`](src/README.md) for the original authors' pipeline description.

## Credit

All credit for the method and the original implementation goes to the upstream
authors. This is a research fork; expect rough edges around the experiment
scripts.
