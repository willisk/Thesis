```
python projectOK/main.py \
-seed=300 \
-n_classes=10 \
-n_dims=20 \
-n_samples_A=1000 \
-n_samples_B=300 \
-n_samples_valid=1000 \
-perturb_strength=1.5 \
-g_modes=12 \
-g_scale_mean=3 \
-g_scale_cov=20 \
-g_mean_shift=1 \
-nn_lr=0.01 \
-nn_steps=300 \
-nn_width=32 \
-nn_depth=4 \
-n_random_projections=32 \
-inv_lr=0.1 \
-inv_steps=1000 \
--nn_resume_train \
--nn_reset_train \
```

# Testing reconstruction methods on high-dimensional Gaussian Mixtures
    
    Hyperparameters:
    n_classes=10 n_dims=20 n_samples_A=1000 n_samples_B=300 n_samples_valid=1000 perturb_strength=1.5 g_modes=12 g_scale_mean=3.0 g_scale_cov=20.0 g_mean_shift=1.0 nn_lr=0.01 nn_steps=300 nn_width=32 nn_depth=4 nn_resume_train=True nn_reset_train=True n_random_projections=32 inv_lr=0.1 inv_steps=1000 seed=300 
    
    No Checkpoint found / Reset.
    Beginning training.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_4_3.png)


    net accuracy: 79.0%
## Method: NN feature
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_4_7.png)


    Results:
    	loss: 0.000
    	l2 reconstruction error: 9.636
    	cross entropy of B: inf
    	nn accuracy: 11.0 %
    	nn validation accuracy: 11.1 %
## Method: NN feature CC
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_4_11.png)


    Results:
    	loss: 5749.976
    	l2 reconstruction error: 30.526
    	cross entropy of B: inf
    	nn accuracy: 40.7 %
    	nn validation accuracy: 32.4 %
## Method: RP
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_4_15.png)


    Results:
    	loss: 0.000
    	l2 reconstruction error: 9.636
    	cross entropy of B: inf
    	nn accuracy: 11.0 %
    	nn validation accuracy: 11.1 %
## Method: RP CC
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_4_19.png)


    Results:
    	loss: 0.485
    	l2 reconstruction error: 9.308
    	cross entropy of B: inf
    	nn accuracy: 29.0 %
    	nn validation accuracy: 30.1 %
    
    Summary
    =======
    
    Data A
    cross entropy: 54.999
    nn accuracy: 79.0 %
    
    perturbed Data B
    cross entropy: inf
    nn accuracy: 11.2 %
    nn accuracy B valid: 10.7 %
    
    method         loss     l2-err  acc   acc(val)  c-entr  
    --------------------------------------------------------
    NN feature     0.00     9.64    0.11  0.11      inf     
    NN feature CC  5749.98  30.53   0.41  0.32      inf     
    RP             0.00     9.64    0.11  0.11      inf     
    RP CC          0.49     9.31    0.29  0.30      inf     

