# Simple Attention

Simple-attention is a Python library that offers a simplified alternative
attention mechanism to the transformer architecture. It aims to address the
complexity of transformers by using an inverse distance weighting approach
instead of scaled dot-product attention. This simplification eliminates the
need for position encoding, making it easier to understand and implement.

The key features of SimpleAttention include:

- **Inverse Distance Weighting**: SimpleAttention assigns attention weights
based on the [inverse of the distance between input
elements](https://en.wikipedia.org/wiki/Inverse_distance_weighting), eliminating
the need for softmax calculations.

- **No Position Encoding**: Unlike regular transformers, SimpleAttention treats
position as a regular feature and does not require additional encoding
techniques such as sinusoidal or rotational encodings.

- **Support for Higher-Dimensional Positions**: SimpleAttention naturally
extends to higher-dimensional positions, such as image patches, enabling it to
handle complex spatial relationships in the input.

While SimpleAttention is still in the experimental phase and not yet
production-grade, initial results show promise. It can serve as a valuable
simplification of the transformer architecture, making it easier to comprehend
and work with.

SimpleAttention is released under the [BSD license](./LICENSE), allowing you to
use it in your projects and modify it as needed.