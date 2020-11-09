python projectOK/main.py \

# Testing reconstruction methods on high-dimensional Gaussian Mixtures
    
    Hyperparameters:
    n_classes=10 n_dims=30 n_samples=1000 perturb_strength=1.5 g_modes=12 g_scale_mean=3.0 g_scale_cov=20.0 g_mean_shift=0.0 nn_lr=0.01 nn_steps=200 nn_width=32 nn_depth=4 nn_resume_train=True nn_reset_train=True n_random_projections=32 inv_lr=0.1 inv_steps=100 seed=333 
    
    No Checkpoint found / Reset.
    Beginning training.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_5_3.png)


    net accuracy: 87.2%
## Method: NN feature
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_5_7.png)


    Results:
    	loss: 0.008
    	l2 reconstruction error: 14.958
    	cross entropy of B: inf
    	nn accuracy: 5.7 %
## Method: NN feature CC
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_5_11.png)


    Results:
    	loss: 164791.125
    	l2 reconstruction error: 169.524
    	cross entropy of B: inf
    	nn accuracy: 13.3 %
## Method: RP
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_5_15.png)


    Results:
    	loss: 0.008
    	l2 reconstruction error: 14.958
    	cross entropy of B: inf
    	nn accuracy: 5.7 %
## Method: RP CC
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_5_19.png)


    Results:
    	loss: 1.603
    	l2 reconstruction error: 30.579
    	cross entropy of B: inf
    	nn accuracy: 16.8 %
    
    Summary
    =======
    
    Data A
    cross entropy: 75.819
    nn accuracy: 87.2 %
    
    perturbed Data B
    cross entropy: inf
    nn accuracy: 11.9 %
    
    method         loss       l2 err  accuracy  cross-entropy  
    -----------------------------------------------------------
    NN feature     0.01       14.96   0.06      inf            
    NN feature CC  164791.12  169.52  0.13      inf            
    RP             0.01       14.96   0.06      inf            
    RP CC          1.60       30.58   0.17      inf            

