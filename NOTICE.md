# Source and licensing notice

The perception, segmentation, storage, retrieval, and model adapters were
exported from the authors' local MEMO implementation. The benchmark runner,
configuration files, dataset adapters, result reporting, tests, and synthetic
example in this release were organized for independent execution.

No third-party model weights or benchmark videos are distributed. Dependency
packages, model weights, and datasets remain subject to their respective terms.
The SAM2 source revision is pinned in requirements.txt; its source is fetched
as a dependency rather than vendored here.

The project code and repository-authored documentation are released under the
MIT License; see LICENSE. The copyright line follows the authors listed in
CITATION.cff. The images in assets/teaser.png and assets/pipeline.png are
rendered from the paper's arXiv v1 source figures (pdf/teaser.pdf and
pdf/framework.pdf), credited to the MEMO authors, and retain the paper's
Creative Commons Attribution 4.0 license. See the [arXiv v1 source](https://arxiv.org/abs/2609.38900v1)
and the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
The only change to the figures is rasterization from PDF to PNG for README
display.

Confirm attribution obligations for any reused third-party code before public
distribution. Model weights and benchmark datasets retain their own terms.
