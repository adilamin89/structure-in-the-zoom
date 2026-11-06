"""How much of the order-averaged topic and construction lift survives matching the number of distinct final tokens per
class, in all four Pythia sizes (App I 'How much is the final token'). Reads the matched arm (llm_lasttoken_<model>.json)
and its size-matched random arm (llm_lasttoken_power_<model>.json); both report the median order-averaged shift per layer.
Per model and partition: the embedding's lift (matched vs size-matched), the share of the lift removed over the first three blocks and
over the second half of the layers (ratio of layer means), and the layers past the embedding where each arm clears its null
(median p <= 0.05).
Out: data_canonical/llm_lasttoken_fractions.json"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = DATA / "llm_lasttoken_fractions.json"
MODELS = ["pythia-160m", "pythia-410m-deduped", "pythia-1b-deduped", "pythia-2.8b-deduped"]
PARTITIONS = {"world_knowledge32": "content", "language_type32": "constructions"}


def main():
    out = {}
    for mk in MODELS:
        matched = json.load(open(DATA / f"llm_lasttoken_{mk}.json"))["batteries"]
        power = json.load(open(DATA / f"llm_lasttoken_power_{mk}.json"))
        for bn, name in PARTITIONS.items():
            m_rows, r_rows = matched[bn]["matched"], power[bn]["random_matched_size"]
            assert [x["layer"] for x in m_rows] == [x["layer"] for x in r_rows], (mk, bn)
            m = np.array([x["dbar_median"] for x in m_rows])
            r = np.array([x["dbar_median"] for x in r_rows])
            pm = np.array([x["p_median"] for x in m_rows])
            pr = np.array([x["p_median"] for x in r_rows])
            n_layers = len(m) - 1                                   # blocks past the embedding
            late = slice(n_layers // 2 + 1, n_layers + 1)
            out[f"{mk}/{name}"] = {
                "m_distinct_last_tokens": matched[bn]["m"],
                "embedding_matched": float(m[0]), "embedding_size_matched": float(r[0]),
                "removed_first3_blocks": float(1 - m[1:4].mean() / r[1:4].mean()),
                "removed_second_half": float(1 - m[late].mean() / r[late].mean()),
                "past_embedding_matched_p05": int((pm[1:] <= 0.05).sum()),
                "past_embedding_size_matched_p05": int((pr[1:] <= 0.05).sum()),
                "n_past_embedding": int(n_layers),
            }
    vals = list(out.values())
    out["summary"] = {
        "embedding_matched_range": [min(v["embedding_matched"] for v in vals), max(v["embedding_matched"] for v in vals)],
        "embedding_size_matched_range": [min(v["embedding_size_matched"] for v in vals), max(v["embedding_size_matched"] for v in vals)],
        "removed_first3_range": [min(v["removed_first3_blocks"] for v in vals), max(v["removed_first3_blocks"] for v in vals)],
        "removed_second_half_range": [min(v["removed_second_half"] for v in vals), max(v["removed_second_half"] for v in vals)],
    }
    json.dump(out, open(OUT, "w"), indent=1)
    for k, v in out.items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v)


if __name__ == "__main__":
    main()
