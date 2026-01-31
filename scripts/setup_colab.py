import os
import sys
import glob

def run_command(command):
    print(f"Running: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        raise Exception(f"Command failed with exit code {exit_code}: {command}")

def main():
    print("Starting Setup...")

    # 1. Install Rust
    print("Installing Rust...")
    run_command("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")

    # Set environment path for rust
    # Note: modifying os.environ affects the current process and child processes started *after* this modification
    os.environ['PATH'] += ":/root/.cargo/bin"
    
    # Verify cargo is accessible
    os.system("cargo --version")

    # 2. Clone the Repo
    print("Cloning Repository...")
    if not os.path.exists("relational-transformer"):
        run_command("git clone -b dev https://github.com/hackdeploy/relational-transformer.git")
    else:
        print("Folder 'relational-transformer' already exists. Skipping clone.")

    os.chdir("relational-transformer")

    # 3. Patch Cargo.toml to support Colab's Python version
    print("Patching Cargo.toml...")
    python_version = f"abi3-py{sys.version_info.major}{sys.version_info.minor}"
    # Using sed to replace the PyO3 feature
    run_command("sed -i 's/abi3-py312/extension-module/g' rustler/Cargo.toml")

    # 4. Install Required Packages
    print("Installing Python packages...")
    run_command("pip install maturin maturin-import-hook uv")
    run_command("pip install torch sentence_transformers wandb einops polars relbench google-cloud-bigquery")

    # 5. Build and install the Rust extension (with Google Drive caching)
    print("Setting up Rust Extension with Drive Caching...")
    
    try:
        from google.colab import drive
        if not os.path.exists("/content/drive"):
            drive.mount('/content/drive')
    except ImportError:
        print("Warning: google.colab module not found. Skipping Google Drive mount.")
        # We continue anyway, hoping the path exists or the user handles it
    
    drive_wheels_dir = "/content/drive/MyDrive/Colab_Data/relational-transformer-wheels"
    run_command(f"mkdir -p '{drive_wheels_dir}'")

    os.chdir("rustler")
    
    # Check for cached wheels
    cached_wheels = glob.glob(f"{drive_wheels_dir}/*.whl")
    
    if cached_wheels:
        print(f"Found cached wheel: {cached_wheels[0]}")
        run_command(f"pip install '{cached_wheels[0]}'")
    else:
        print("No cached wheel found. Building Rust extension...")
        run_command("maturin build --release")
        
        # Install the generated wheel
        # We rely on shell expansion for the wildcard, so we use os.system directly or ensure shell=True behavior
        run_command("pip install target/wheels/*.whl")
        
        # Copy to Drive
        print(f"Caching wheel to {drive_wheels_dir}...")
        run_command(f"cp target/wheels/*.whl '{drive_wheels_dir}/'")

    os.chdir("..")
    print("Setup complete! You can now import rt.")

if __name__ == "__main__":
    main()
