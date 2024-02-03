# BTEval

[![PyPI - Version](https://img.shields.io/pypi/v/bteval.svg)](https://pypi.org/project/bteval)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bteval.svg)](https://pypi.org/project/bteval)

-----

BTEval is a Python library for measuring the robustness of natural language understanding models to speech recognition errors.
It implements the family of `R*` robustness measures defined in [Back Transcription as a Method for Evaluating Robustness of Natural Language Understanding Models to Speech Recognition Errors](https://aclanthology.org/2023.emnlp-main.724) (Kubis et al., EMNLP 2023).

## Installation

```console
pip install bteval
```

## Usage

```python
from bteval import r1_score

y_true = ["Inform", "Request", "Inform"]
y_before = ["Inform", "Request", "Request"]
y_after = ["Inform", "Confirm", "Confirm"]

r1_score(y_true, y_before, y_after)
```

## Citing

If you use bteval for your research, please cite the following paper:

Marek Kubis, Paweł Skórzewski, Marcin Sowański, and Tomasz Zietkiewicz. 2023. [Back Transcription as a Method for Evaluating Robustness of Natural Language Understanding Models to Speech Recognition Errors](https://aclanthology.org/2023.emnlp-main.724). In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 11824–11835, Singapore. Association for Computational Linguistics.

```bibtex
@InProceedings{kubis-etal-2023-back,
  title        = "{Back Transcription as a Method for Evaluating Robustness of Natural Language Understanding Models to Speech Recognition Errors}",
  author       = "Kubis, Marek and Skórzewski, Paweł and Sowański, Marcin and Ziętkiewicz, Tomasz",
  booktitle    = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  month        = dec,
  year         = "2023",
  address      = "Singapore",
  publisher    = "Association for Computational Linguistics",
  URL          = "https://aclanthology.org/2023.emnlp-main.724",
  doi          = "10.18653/v1/2023.emnlp-main.724",
  pages        = "11824--11835",
}
```
