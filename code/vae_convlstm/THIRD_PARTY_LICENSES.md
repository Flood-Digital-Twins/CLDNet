# Third-party code in this folder

Parts of the VAE–ConvLSTM baseline are adapted from the open-source projects below, all released under the MIT
License. Their copyright notices are reproduced here as that license requires.

| Files | Source | Copyright |
|---|---|---|
| `ldm_ae/ae.py`, `ldm_ae/model.py`, `ldm_ae/attention.py`, `ldm_ae/distributions.py`, `ldm_ae/util.py`; the encoder and decoder blocks in `vae_texas.py` | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | Copyright (c) 2022 Machine Vision and Learning Group, LMU Munich |
| `ldm_ae/quantize.py` | [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) | Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer |
| `ldm_ae/convlstm.py` (modified to accept and return the hidden state) | [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch) | Copyright (c) 2017 Andrea Palazzi |
| `soap.py` | [nikhilvyas/SOAP](https://github.com/nikhilvyas/SOAP) | Copyright (c) 2024 Nikhil Vyas |

## MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
