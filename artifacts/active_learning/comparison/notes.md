# AL strategy comparison — auto notes
Primary metric: `material_macro_f1` on val_gold.
Reference (B=0, train_seed only): val=0.7030, test=0.5818

## Budget B=50
- **coreset** val_material=0.7876 (Δvs_seed=+0.0846, Δvs_random=+0.0944) test_material=0.6434 redund=0.795
- **random** val_material=0.6933 (Δvs_seed=-0.0098, Δvs_random=+nan) test_material=0.6546 redund=0.734
- **least_confidence** val_material=0.6859 (Δvs_seed=-0.0171, Δvs_random=-0.0073) test_material=0.7067 redund=0.739
- **hybrid** val_material=nan (Δvs_seed=+nan, Δvs_random=+nan) test_material=nan redund=0.783

## Budget B=100
- **random** val_material=0.7408 (Δvs_seed=+0.0378, Δvs_random=+nan) test_material=0.6393 redund=0.764
- **coreset** val_material=0.6998 (Δvs_seed=-0.0032, Δvs_random=-0.0410) test_material=0.7068 redund=0.774
- **least_confidence** val_material=0.6819 (Δvs_seed=-0.0211, Δvs_random=-0.0589) test_material=0.7331 redund=0.736
- **hybrid** val_material=nan (Δvs_seed=+nan, Δvs_random=+nan) test_material=nan redund=0.770
