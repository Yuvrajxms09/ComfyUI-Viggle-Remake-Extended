# H3 resolution alignment

H3 canvas width and height must be multiples of 32. The extended node keeps 32-pixel choices in its UI and treats short edge as a maximum: unaligned API requests round down to the largest valid 32-pixel short edge that does not exceed the request. Values outside the installed ComfyUI range still produce a range error.

The production resolution buckets use their labels as maximum short edges and round down to the largest aligned value within the cap: `480p` → 480, `720p` → 704, and `1080p` → 1056. For the landscape 16:9-class `1080p` canvas, the target is 1856×1056. The next aligned size, 1920×1088, exceeds the 1080 maximum. A raw API request of `short_edge=1080` follows the same rule and becomes 1056.

## Recorded exact-1080 failure

The 2026-11-12 two-second portrait probe requested a 1080 short edge and failed in ComfyUI's `SamplerCustomAdvanced` with a shape-reshape `RuntimeError`. The trace reported a spatial latent dimension of 67 being reshaped into 33 groups of 2. The VAE's 16-pixel spatial stride maps a 1080-pixel side to 67 latent cells; that odd size cannot be divided into the transformer's 2-cell spatial patches. This was a shape/alignment failure during sampling, not an out-of-memory error or a failure to load the LoRA.

The rounded-down 1056 size gives 66 latent cells on the short side, so it should avoid that specific odd-size reshape error while honoring the requested cap. The exact case has not yet been rerun at 1056, so other sampler, memory, or workflow failures remain possible.
