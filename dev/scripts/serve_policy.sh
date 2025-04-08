# uv run scripts/serve_policy.py \
    # --port 8000 \
    # policy:checkpoint \
    # --policy.config=pi0_fast_libero \
    # --policy.dir=checkpoints/pi0_fast_libero/pi0_fast_libero/29999

uv run scripts/serve_policy.py \
    --port 8000 \
    --env LIBERO