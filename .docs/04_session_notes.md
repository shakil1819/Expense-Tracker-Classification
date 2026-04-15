# Session Notes

- Repository instructions require Python, git-traceable work, `.docs/` notes, and practical validation.
- `description.md` was already modified before this session and was left untouched.
- `uv` was installed locally and used to manage dependencies.
- The codebase was restructured from `peakflo/` into `src/` with `feature_engineering_pipeline.py`, `training_pipeline.py`, and `inference_pipeline.py`.
- The final implementation favors sparse linear text models over heavier approaches because the dataset is small and text-heavy.
- New diagnostics were added for overfitting, underfitting, learning curves, validation curves, and balanced unseen evaluation.
