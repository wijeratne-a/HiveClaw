Place the release ``pheromoned`` binary here before building a wheel (macOS arm64):

  cargo build --release -p hiveclaw-daemon
  cp ../../../../target/release/pheromoned ./pheromoned

The file ``pheromoned`` is gitignored. CI copies it automatically before maturin build.
