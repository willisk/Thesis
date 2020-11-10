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
-g_mean_shift=2 \
-nn_lr=0.01 \
-nn_steps=100 \
-nn_width=32 \
-nn_depth=4 \
-n_random_projections=32 \
-inv_lr=0.1 \
-inv_steps=300 \
--nn_verifier \

# --nn_reset_train \
# --nn_resume_train \
```

    
    Hyperparameters:
    n_classes=10
    n_dims=20
    n_samples_A=1000
    n_samples_B=300
    n_samples_valid=1000
    perturb_strength=1.5
    g_modes=12
    g_scale_mean=3.0
    g_scale_cov=20.0
    g_mean_shift=2.0
    nn_lr=0.01
    nn_steps=100
    nn_width=32
    nn_depth=4
    nn_resume_train=True
    nn_reset_train=True
    nn_verifier=True
    n_random_projections=32
    inv_lr=0.1
    inv_steps=300
    seed=300
    
    Running on 'cpu'
    No Checkpoint found / Reset.
    Beginning training.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_3.png)


    net accuracy: 88.7%
    No Checkpoint found / Reset.
    Beginning training.





    
    verifier net accuracy: 89.2%
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_9.png)


    Results:
    	loss: 0.000
    	l2 reconstruction error: 13.280
    	cross entropy of B: inf
    	nn accuracy: 13.2 %
    	nn validation set accuracy: 13.7 %
    	nn verifier accuracy: 13.1 %
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_13.png)


    Results:
    	loss: 13050.233
    	l2 reconstruction error: 76.772
    	cross entropy of B: inf
    	nn accuracy: 28.0 %
    	nn validation set accuracy: 26.1 %
    	nn verifier accuracy: 15.7 %
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_17.png)


    Results:
    	loss: 0.000
    	l2 reconstruction error: 13.280
    	cross entropy of B: inf
    	nn accuracy: 13.2 %
    	nn validation set accuracy: 13.7 %
    	nn verifier accuracy: 13.1 %
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_21.png)


    Results:
    	loss: 0.939
    	l2 reconstruction error: 11.140
    	cross entropy of B: inf
    	nn accuracy: 39.0 %
    	nn validation set accuracy: 38.7 %
    	nn verifier accuracy: 41.8 %
    
    =========
    
    Data A
    cross entropy: 54.999
    nn accuracy: 88.7 %
    
    perturbed Data B
    cross entropy: inf
    nn accuracy: 12.9 %
    nn accuracy B valid: 11.7 %
    nn verifier accuracy: 12.0 %
    
    method         loss      l2-err  acc   acc(val)  acc(ver)  c-entr  
    -------------------------------------------------------------------
    NN feature     0.00      13.28   0.13  0.14      0.13      inf     
    NN feature CC  13050.23  76.77   0.28  0.26      0.16      inf     
    RP             0.00      13.28   0.13  0.14      0.13      inf     
    RP CC          0.94      11.14   0.39  0.39      0.42      inf     


# TESTS


