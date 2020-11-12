

```
python projectOK/main.py \
-seed=300 \
-n_classes=10 \
-n_dims=20 \
-n_samples_A=5000 \
-n_samples_B=1000 \
-n_samples_valid=5000 \
-perturb_strength=1.5 \
-g_modes=12 \
-g_scale_mean=3 \
-g_scale_cov=20 \
-g_mean_shift=2 \
-nn_lr=0.01 \
-nn_steps=100 \
-nn_width=32 \
-nn_depth=4 \
-n_random_projections=128 \
-inv_lr=0.1 \
-inv_steps=500 \
--nn_verifier \

# --nn_reset_train \
# --nn_resume_train \
```

# Testing reconstruction methods on high-dimensional Gaussian Mixtures
    
    Hyperparameters:
    n_classes=10
    n_dims=20
    n_samples_A=5000
    n_samples_B=1000
    n_samples_valid=5000
    perturb_strength=1.5
    g_modes=12
    g_scale_mean=3.0
    g_scale_cov=20.0
    g_mean_shift=2.0
    nn_lr=0.01
    nn_steps=100
    nn_width=32
    nn_depth=4
    nn_resume_train=False
    nn_reset_train=False
    nn_verifier=True
    n_random_projections=128
    inv_lr=0.1
    inv_steps=500
    seed=300
    
    Running on 'cpu'


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)


    Training Checkpoint restored: /content/Thesis/models/net_GMM_20-32-32-32-32-10.pt
    net accuracy: 79.3%
    Training Checkpoint restored: /content/Thesis/models/net_GMM_20-32-32-32-32-10_verifier.pt
    verifier net accuracy: 78.6%
## Method: NN
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_5.png)


    Results:
    	loss: -0.000
    	l2 reconstruction error: 13.676
    	cross entropy of B: inf
    	nn accuracy: 14.8 %
    	nn validation set accuracy: 14.3 %
    	nn verifier accuracy: 14.1 %
## Method: NN CC
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_9.png)


    Results:
    	loss: 15.503
    	l2 reconstruction error: 58.302
    	cross entropy of B: inf
    	nn accuracy: 56.2 %
    	nn validation set accuracy: 54.6 %
    	nn verifier accuracy: 32.7 %
## Method: NN ALL
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_13.png)


    Results:
    	loss: -0.000
    	l2 reconstruction error: 13.676
    	cross entropy of B: inf
    	nn accuracy: 14.8 %
    	nn validation set accuracy: 14.3 %
    	nn verifier accuracy: 14.1 %
## Method: NN ALL CC
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_17.png)


    Results:
    	loss: 179.166
    	l2 reconstruction error: 17.636
    	cross entropy of B: inf
    	nn accuracy: 71.4 %
    	nn validation set accuracy: 71.3 %
    	nn verifier accuracy: 66.6 %
## Method: RP
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_21.png)


    Results:
    	loss: -0.000
    	l2 reconstruction error: 13.676
    	cross entropy of B: inf
    	nn accuracy: 14.8 %
    	nn validation set accuracy: 14.3 %
    	nn verifier accuracy: 14.1 %
## Method: RP CC
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_25.png)


    Results:
    	loss: 0.698
    	l2 reconstruction error: 6.610
    	cross entropy of B: inf
    	nn accuracy: 68.8 %
    	nn validation set accuracy: 70.0 %
    	nn verifier accuracy: 69.3 %
## Method: RP ReLU
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_29.png)


    Results:
    	loss: -0.000
    	l2 reconstruction error: 13.676
    	cross entropy of B: inf
    	nn accuracy: 14.8 %
    	nn validation set accuracy: 14.3 %
    	nn verifier accuracy: 14.1 %
## Method: RP ReLU CC
    Beginning Inversion.





    


    /content/Thesis/utility.py:413: UserWarning: Attempting to set identical bottom == top == 4664.62353515625 results in singular transformations; automatically expanding.
      plt.gca().set_ylim([y_min - buffer, y_max + buffer])



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_34.png)


    Results:
    	loss: nan
    	l2 reconstruction error: nan
    	cross entropy of B: nan
    	nn accuracy: 10.0 %
    	nn validation set accuracy: 10.0 %
    	nn verifier accuracy: 10.0 %
## Method: combined
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_7_38.png)


    Results:
    	loss: 94.273
    	l2 reconstruction error: 16.788
    	cross entropy of B: inf
    	nn accuracy: 71.6 %
    	nn validation set accuracy: 71.5 %
    	nn verifier accuracy: 67.6 %
    
# Summary
    =========
    
    Data A
    cross entropy: 54.994
    nn accuracy: 79.3 %
    
    perturbed Data B
    cross entropy: inf
    nn accuracy: 12.3 %
    nn accuracy B valid: 12.2 %
    nn verifier accuracy: 12.3 %
    
    method      loss    l2-err  acc   acc(val)  acc(ver)  c-entr  
    --------------------------------------------------------------
    NN          -0.00   13.68   0.15  0.14      0.14      inf     
    NN CC       15.50   58.30   0.56  0.55      0.33      inf     
    NN ALL      -0.00   13.68   0.15  0.14      0.14      inf     
    NN ALL CC   179.17  17.64   0.71  0.71      0.67      inf     
    RP          -0.00   13.68   0.15  0.14      0.14      inf     
    RP CC       0.70    6.61    0.69  0.70      0.69      inf     
    RP ReLU     -0.00   13.68   0.15  0.14      0.14      inf     
    RP ReLU CC  nan     nan     0.10  0.10      0.10      nan     
    combined    94.27   16.79   0.72  0.71      0.68      inf     


