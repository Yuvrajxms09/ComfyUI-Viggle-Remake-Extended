# H3 resolution alignment

H3 canvas width and height must be multiples of 32. The extended node keeps 32-pixel choices in its UI and rounds unaligned API requests to the nearest valid short edge. Values outside the installed ComfyUI range still produce a range error.

The runner's `1080p` bucket maps to 1088, the closest 32-aligned short edge to 1080. A landscape 16:9-class canvas targets 1920×1088; a portrait canvas targets 1088×1920. The neighboring lower aligned short edge is 1056.

## Recorded exact-1080 failure

The 2026-11-12 two-second portrait probe requested a 1080 short edge and failed in ComfyUI's `SamplerCustomAdvanced` with a shape-reshape `RuntimeError`. The trace reported a spatial latent dimension of 67 being reshaped into 33 groups of 2. The VAE's 16-pixel spatial stride maps a 1080-pixel side to 67 latent cells; that odd size cannot be divided into the transformer's 2-cell spatial patches. This was a shape/alignment failure during sampling, not an out-of-memory error or a failure to load the LoRA.

Rounding to 1088 makes that side 68 latent cells, which is divisible by 2. This should prevent the same reshape error. The exact case has not yet been rerun at 1088, so other sampler, memory, or workflow failures remain possible.
