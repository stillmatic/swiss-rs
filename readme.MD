## swiss

simulation of swiss tournament in Rust

mostly an exercise to learn Rust! it's extremely fast though.

```
▶ time cargo run --release 64 6 1000
    Finished release [optimized] target(s) in 0.01s
     Running `target/release/swiss 64 6 1000`
cargo run --release 64 6 1000  1.58s user 0.02s system 99% cpu 1.603 total

▶ time cargo run --release 256 16 1000
    Finished release [optimized] target(s) in 0.01s
     Running `target/release/swiss 256 16 1000`
cargo run --release 256 16 1000  63.10s user 0.17s system 99% cpu 1:03.28 total
```

the bottleneck is computing the maximum weight matching (`O(n^3)` where n = # of nodes). so at least there isn't hidden hyperlinear scaling or whatever.