# Incipit trainer

A minimal CPU-based NNUE trainer, used to help bootstrap Incipit's training from its own self-play data. It is multi-threaded and uses only the Rust standard library.

In the near future, model training will move to [Bullet](https://github.com/jw1912/bullet).

## Usage

```
cargo run --release -- <out> <hidden> <epochs> <lr> <lambda> <threads> [options] <data files...>
```

Options: `--cosine`, `--mirror`, `--kb <n>`, `--valfile <file>`, `--init <file>`.
