# Tokenization Code

This folder contains only the tokenization code required by the downstream source families.

## Included Tokenizers

- `wl.py`: loads and applies the WL tokenizer used by the all-domain WL and domain-dependent WL source models.
- `shortest_path.py`: loads and applies the shortest-path tokenizer used by the domain-dependent LSTM source family.
- `multidomain.py`: loads the all-domain WL block-union tokenizer manifest.
- `base.py`: shared tokenizer interface.
- `factory.py`: small factory restricted to WL and shortest-path tokenizers.

I removed SimHash, GraphBPE, and random tokenizers from this folder because none of the four downstream source families uses them.
