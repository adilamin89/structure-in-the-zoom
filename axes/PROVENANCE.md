# Prompt and stimulus provenance

Every prompt in `axes/` and every stimulus stored in a `data_canonical/` artifact is listed here with its
origin and license. Code in this repository is MIT (see `LICENSE`); the items below carry their own terms.

## Axes composed for the paper (released under CC BY 4.0)

| axis | classes | how it was made |
|---|---|---|
| `world_knowledge.json` | 8 domains x 16 prompts | composed for the paper by the author, fixed before any measurement |
| `language_type.json` | 8 construction types x 16 | same; topics deliberately mixed inside each class |
| `language_type.strata.json` | topic stratum per prompt | assigned by hand from the prompt text alone, blind to any measured value (the nuisance-preserving null of App G) |
| `ethical.json` | 8 ethical concepts x 16 | same as world_knowledge; the concept list is inspired by, not identical to, the moral-foundations taxonomy |
| `compass.json`, `clock.json` (+ `.strata.json`) | 8 classes x 16 shared carriers each | the planted C8 axes of Sec 8.5: sixteen carrier sentences composed for the paper with only the class token rotated; the sidecar holds the carrier id per prompt (the carrier-stratified null) |

The three composed axes and the strata are the paper's own instrument design. Style features of the
prompts (short declarative carriers, matched final tokens) are part of that design and are stated in the paper.

## Axes mined from public benchmarks (verbatim items; each file inherits its source license)

| axis | source | license | citation |
|---|---|---|---|
| `ethics_benchmark.json` | ETHICS scenario sentences, four normative domains x 32, reformatted as prompt-plus-completion text | MIT | Hendrycks et al. 2021 (via the `wassname/ethics_expression_preferences` parquet mirror) |
| `tqa_category.json` | TruthfulQA questions, grouped by the release's category field | Apache-2.0 | Lin, Hilton, Evans 2022 |
| `hs_activity.json` | HellaSwag contexts (WikiHow-derived), grouped by activity label | MIT | Zellers et al. 2019 |
| `arc_topic.json` | ARC questions, grouped by science topic | CC BY-SA 4.0 (this derivative file is therefore CC BY-SA 4.0) | Clark et al. 2018 |

## Stimuli stored inside result artifacts

| artifact | stored text | source | license |
|---|---|---|---|
| `run46_ethics_benchmark_axis.json` | 128 ETHICS scenario sentences (4 domains x 32), reformatted as prompt-plus-completion text | ETHICS (Hendrycks et al. 2021), via the `wassname/ethics_expression_preferences` parquet mirror | MIT |
| `run43_baroni_complexity.json`, `run43b_baroni_64pairs.json`, `run43c_baroni_crossmodel.json` | the minimal-pair prompts used | Baroni et al. 2026 release, github.com/franfranz/syntactic_complexity_in_LLMs | no license file in the release; redistributed here as research data with attribution, contact the authors before any other reuse |
| `run42_blimp_battery.json` | none (BLiMP is loaded at run time from `nyu-mll/blimp`) | Warstadt et al. 2020 | CC BY 4.0 |

Reuse of any benchmark-derived file must carry the source's attribution; the ARC-derived axis must stay
share-alike. If you build your own axis with `theta-zoom axis`, the same applies to the dataset you draw from.
