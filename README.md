# `adagram`

A Rust reimplementation of the Adaptive Skip-gram model for Word Sense
Induction, following the algorithm described in:

Sergey Bartunov, Dmitry Kondrashkin, Anton Osokin, and Dmitry P. Vetrov. 2016.
*Breaking Sticks and Ambiguities with Adaptive Skip-gram*. In *Proceedings of the
19th International Conference on Artificial Intelligence and Statistics (AISTATS)*,
PMLR 51:130-138. https://proceedings.mlr.press/v51/bartunov16.html

The model extends skip-gram by replacing a single embedding per word with
multiple sense-specific prototypes. A Dirichlet-process prior controls the
effective number of senses per word. Training uses hierarchical softmax and
Hogwild-style parallelism. Storage formats are compatible with the
original implementation.

The programs implemented here serve as the basic building block for trending sense detection provided by the [`sensetrends`](https://github.com/ondra/sensetrends) repository.

## Installation

### Prebuilt binaries

Prebuilt statically linked binaries for x86_64 Linux are available in the
`bin/` directory. They have no external dependencies. The binaries use efficient code paths for different SIMD targets, so a native build will likely not be much faster.

### Install from source

```
cargo install --git https://github.com/ondra/adagram
```

### Build from source

The project depends on [`corp`](https://github.com/ondra/corp) and
[`slope`](https://github.com/ondra/slope), which are fetched automatically
from GitHub by Cargo.

```
git clone https://github.com/ondra/adagram
cd adagram
cargo build --release
```

The `.cargo/config.toml` enables `-C target-cpu=native` by default for best
performance on the build machine. Binaries will be in `target/release/`.

### Building static binaries

To build fully static binaries for distribution, uncomment the musl config in
`.cargo/config.toml`:

```toml
[build]
rustflags = ["-C", "target-feature=+crt-static", "-C", "target-cpu=native"]
target = "x86_64-unknown-linux-musl"
```

This requires the musl target (`rustup target add x86_64-unknown-linux-musl`).
Binaries will be in `target/x86_64-unknown-linux-musl/release/`.

## Corpus input

The training and inference tools read corpora through the
[`corp`](https://github.com/ondra/corp) crate, which handles the binary
format used by [Manatee](https://nlp.fi.muni.cz/trac/noske) /
[Sketch Engine](https://www.sketchengine.eu/). Corpora already indexed by
Manatee can be used directly.

To use plain vertical text (one token per line with tab-separated attributes),
compile it first with `encodevert` from the `corp` package:

```bash
# compile the vertical text into binary corpus format
encodevert -c corpus.conf

# build reverse index (needed for concordancing tools)
mkrev corpus.conf word
```

See the [`corp` README](https://github.com/ondra/corp) for details on the
configuration file format, corpus compilation, and the available tools.

## Quick example

```bash
# train a model on the "lempos" attribute of a compiled corpus
learn ./corpus.conf lempos model.adagram \
    --dim 64 --alpha 0.15 --window 10

# show nearest neighbors for each induced sense
echo bank-n | nearest model.adagram --compact

# extract representative concordance lines per sense
echo bank-n | senseconc ./corpus.conf lempos word model.adagram

# compute sense-by-epoch frequency distributions
echo bank-n | sensefreqs ./corpus.conf model.adagram lempos doc.month
```

## Tools

### learn

Train an Adaptive Skip-gram model on a compiled corpus.

```
learn [OPTIONS] <CORPUS> <ATTR> <MODEL_FILE>
```

Options:
- `--window N` — context window size (default: 4)
- `--dim N` — embedding dimensionality (default: 100)
- `--prototypes N` — maximum senses per word (default: 5)
- `--alpha F` — Dirichlet process concentration parameter (default: 0.1)
- `--min-freq N` — minimum word frequency (default: 20)
- `--epochs N` — training passes (default: 1)
- `--subsample F` — subsampling threshold (default: inf)
- `--context-cut` — randomly reduce context size

Example:

```
learn corpus.conf lempos model.adagram \
    --dim 64 --alpha 0.15 --window 10
```

### sensefreqs

Assign word occurrences to senses and aggregate into per-epoch frequency
distributions. Reads headwords (one per line) from stdin.

```
sensefreqs [OPTIONS] <CORPUS> <POSATTR> <DIAATTR> <MODEL>
```

Options:
- `--window N` - context window size (inferred from model path if omitted)
- `--sense-threshold F` - minimum a priori sense probability (default: 0.001)
- `--epoch-limit F` - minimum norm for an epoch to be included (default: 0.15)
- `--distrib` - output soft probability distributions instead of hard counts
- `--nthreads N` - number of threads

Example:

```
echo bank-n | sensefreqs corpus.conf model.adagram word doc.month
```

Output: tab-separated with columns `hw`, `epoch`, `s0` .. `sK`, `norm`.

### senseconc

Extract representative concordance lines for each sense of a headword, sorted
by disambiguation confidence. Reads headwords from stdin.

```
senseconc [OPTIONS] <CORPUS> <POSATTR> <WORD> <MODEL>
```

Options:
- `--window N` — context window size (inferred from model path if omitted)
- `--conctokens N` — tokens per concordance line (default: 25)
- `--sense-threshold F` — minimum a priori sense probability (default: 0.001)

Example:

```
echo bank-n | senseconc corpus.conf lempos word model.adagram
```

Output: tab-separated with columns `hw`, `sn`, `prob`, `pos`, `lctx`, `kw`, `rctx`.

### nearest

List nearest neighbors in the embedding space for each induced sense. Reads
headwords from stdin.

```
nearest [OPTIONS] <MODEL>
```

Options:
- `--neighbors N` — number of nearest neighbors (default: 10)
- `--minfreq N` — minimum frequency of candidates (default: 5)
- `--minprob F` — minimum a priori sense probability (default: 0.001)
- `--compact` — one line per sense, neighbors space-separated (default)

Example:

```
echo bank-n | nearest model.adagram --compact
```

### desamb

Disambiguate headword+context pairs. Reads tab-separated input from stdin
(`head<TAB>context`).

```
desamb [OPTIONS] <MODEL>
```

Options:
- `--window N` — context tokens on each side (inferred from model path if omitted; 0 = full context)
- `--mirror-input` — append result to input line
- `--print-probs` — output probability distribution over all senses
- `--print-nsenses` — output number of active senses
- `--print-header` — write column names as first output line
- `--skip-columns N` — skip N input columns from left (default: 0)

Example:

```
printf 'bank-n\tcentral-j rate-n\n' \
    | desamb model.adagram --print-probs --mirror-input
```

## Notes

- `learn` saves models in a format compatible with the original AdaGram
  implementation, so models can be used interchangeably.
- Corpus attribute names (`lempos`, `doc.year`, etc.) depend on your corpus
  configuration.

## License

GPL-3.0. See [LICENSE](LICENSE).
