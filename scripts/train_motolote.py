from rt.main import main

if __name__ == "__main__":
    main(
        project="motolote",
        eval_splits=["val"],
        eval_freq=500,
        eval_pow2=False,
        max_eval_steps=20,
        load_ckpt_path=None,
        save_ckpt_dir="ckpts/motolote",
        compile_=True,
        seed=0,
        # data
        train_tasks=[("motolote", "listing_model_matches", "label", [])],
        eval_tasks=[("motolote", "listing_model_matches", "label", [])],
        batch_size=32,
        num_workers=2,
        max_bfs_width=256,
        # optimization
        lr=1e-3,
        wd=0.1,
        lr_schedule=True,
        max_grad_norm=1.0,
        max_steps=5_000,
        # model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
    )
