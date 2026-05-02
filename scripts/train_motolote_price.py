from rt.main import main

if __name__ == "__main__":
    main(
        project="motolote",
        eval_splits=["val", "test"],
        eval_freq=500,
        eval_pow2=False,
        max_eval_steps=50,
        load_ckpt_path="/content/drive/MyDrive/Colab_Data/relational-transformer-checkpoints/contd-pretrain_rel-avito_ad-ctr.pt",
        save_ckpt_dir="ckpts/motolote_price",
        compile_=True,
        seed=0,
        # data
        train_tasks=[("motolote", "listing_prices", "log_price", [])],
        eval_tasks=[("motolote", "listing_prices", "log_price", [])],
        batch_size=32,
        num_workers=2,
        max_bfs_width=256,
        # optimization
        lr=1e-5,  # fine-tuning from pretrained checkpoint
        wd=0.0,
        lr_schedule=False,
        max_grad_norm=1.0,
        max_steps=10_000,
        # model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
    )
