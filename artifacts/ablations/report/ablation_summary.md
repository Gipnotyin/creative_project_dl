# Ablation Study

Best experiment by validation macro-F1: `ablation_cpu_stretch`

| experiment                    | preprocess | class_weights | image_size | epochs | best_epoch | val_mean_macro_f1 |
|-------------------------------|------------|---------------|------------|--------|------------|-------------------|
| ablation_cpu_stretch          | stretch    | True          | 224        | 4      | 4          | 0.7676            |
| ablation_cpu_no_class_weights | pad        | False         | 224        | 4      | 4          | 0.7331            |
| ablation_cpu_base             | pad        | True          | 224        | 4      | 4          | 0.7021            |
