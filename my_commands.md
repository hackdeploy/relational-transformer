# Convert to RKYV files to json
```bash
cd rustler
pixi run cargo run --release --bin convert-file -- "G:\My Drive\Colab_Data\scratch\pre\rel-f1"
```

# Compiling Rustler in Windows to run in Colab using WSL
```bash
wsl --list

wsl
# Your Windows C: drive is mounted at /mnt/c/
cd /mnt/c/Users/User/source/repos/relational-transformer/rustler

# Build the wheel for Linux
maturin build --release

```