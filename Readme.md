# RECONSTRUCTION

## CIFAR10


```
# r_distort=0.2

!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=CIFAR10 \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=1024 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=100 \
-r_distort_level=0.2 \
-r_block_width=16 \
-r_block_depth=4 \
--plot_ideal \
-show_after=20 \
-seed=1 \

# -size_A=-1 \
# -size_B=512 \
# -batch_size=256 \
# -seed=23456

# --reset_stats \
```


```
# pretty samples
# r_distort=0.1

!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=CIFAR10 \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=1024 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=100 \
-r_distort_level=0.1 \
--plot_ideal \
-show_after=20 \
-seed=1
```

    Already up to date.
    HEAD is now at 8b1cf4e depth
# Testing reconstruction methods
# on CIFAR10
    Hyperparameters:
    dataset=CIFAR10
    seed=1
    nn_lr=0.01
    nn_steps=100
    batch_size=128
    n_random_projections=512
    inv_lr=0.1
    inv_steps=100
    f_reg=0.0
    f_crit=1.0
    f_stats=100.0
    size_A=1024
    size_B=512
    show_after=20
    r_distort_level=0.1
    r_block_depth=4
    r_block_width=4
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=False
    plot_ideal=True
    scale_each=False 
    
    Running on 'cuda'
    
    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to /content/Thesis/data/cifar-10-python.tar.gz






    Extracting /content/Thesis/data/cifar-10-python.tar.gz to /content/Thesis/data
    Files already downloaded and verified
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet34.pt
    net accuracy: 96.6%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet50.pt
    verifier net accuracy: 78.6%
    
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-+-RP-CC-512.pt.
    
    ground truth:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_0.png)


    
    
    distorted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_1.png)


    
    
    
    
## Method: CRITERION
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_0.png)


    


     20%|██        |20.0/100 [01:04<04:16, 3.20s/epoch, accuracy=0.852, ideal=0.427, loss=0.642, psnr=19.2, |grad|=1.58]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_1.png)


    


     40%|████      |40.0/100 [02:08<03:12, 3.21s/epoch, accuracy=0.93, ideal=0.364, loss=0.437, psnr=18.9, |grad|=1.08]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_2.png)


    


     60%|██████    |60.0/100 [03:12<02:08, 3.20s/epoch, accuracy=0.922, ideal=0.428, loss=0.522, psnr=20.6, |grad|=1.29]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_3.png)


    


     80%|████████  |80.0/100 [04:16<01:04, 3.20s/epoch, accuracy=0.961, ideal=0.369, loss=0.396, psnr=20.4, |grad|=0.7]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_4.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.20s/epoch, accuracy=0.992, ideal=0.342, loss=0.353, psnr=21.3, |grad|=1.21]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_5.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.21s/epoch, accuracy=0.992, ideal=0.342, loss=0.353, psnr=21.3, |grad|=1.21]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_7.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_CRITERION_8.png)


    
    Results:
    	loss: 1.540
    	average PSNR: 18.419 | (distorted: 23.959)
    	rel. l2 reconstruction error: 22.265 | (distorted: 7.058)
    	nn accuracy: 95.1 %
    	nn validation set accuracy: 82.6 %
    	nn verifier accuracy: 70.3 %
    
    
    
## Method: NN
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_0.png)


    


     20%|██        |20.0/100 [01:03<04:15, 3.19s/epoch, accuracy=0.828, ideal=1.34, loss=1.69, psnr=16.3, |grad|=2.99]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_1.png)


    


     40%|████      |40.0/100 [02:08<03:12, 3.20s/epoch, accuracy=0.844, ideal=1.37, loss=1.36, psnr=19.4, |grad|=3.33]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_2.png)


    


     60%|██████    |60.0/100 [03:12<02:08, 3.20s/epoch, accuracy=0.875, ideal=1.13, loss=1.26, psnr=18.3, |grad|=4.35]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_3.png)


    


     80%|████████  |80.0/100 [04:16<01:03, 3.19s/epoch, accuracy=0.883, ideal=1.46, loss=1.28, psnr=19.6, |grad|=2.33]  

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_4.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.19s/epoch, accuracy=0.852, ideal=1.34, loss=1.23, psnr=15.7, |grad|=1.85] 

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_5.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.21s/epoch, accuracy=0.852, ideal=1.34, loss=1.23, psnr=15.7, |grad|=1.85]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_8.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_9.png)


    
    Results:
    	loss: 4.628
    	average PSNR: 16.918 | (distorted: 23.884)
    	rel. l2 reconstruction error: 16.275 | (distorted: 7.058)
    	nn accuracy: 89.5 %
    	nn validation set accuracy: 76.4 %
    	nn verifier accuracy: 63.3 %
    
    
    
## Method: NN CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_0.png)


    


     20%|██        |20.0/100 [01:04<04:18, 3.24s/epoch, accuracy=0.898, ideal=1, loss=1.42, psnr=18.7, |grad|=3.29]    

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_1.png)


    


     40%|████      |40.0/100 [02:09<03:13, 3.23s/epoch, accuracy=0.898, ideal=0.978, loss=1.44, psnr=20.1, |grad|=4.65]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_2.png)


    


     60%|██████    |60.0/100 [03:14<02:09, 3.23s/epoch, accuracy=0.938, ideal=1.23, loss=1.4, psnr=19.7, |grad|=3.96]  

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_3.png)


    


     80%|████████  |80.0/100 [04:18<01:04, 3.23s/epoch, accuracy=0.93, ideal=1.18, loss=1.42, psnr=19, |grad|=4.92]   

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_4.png)


    


    100%|██████████|100.0/100 [05:23<00:00, 3.23s/epoch, accuracy=0.922, ideal=0.942, loss=1.36, psnr=21.9, |grad|=5.52]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_5.png)


    


    100%|██████████|100.0/100 [05:23<00:00, 3.24s/epoch, accuracy=0.922, ideal=0.942, loss=1.36, psnr=21.9, |grad|=5.52]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_8.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_CC_9.png)


    
    Results:
    	loss: 4.580
    	average PSNR: 22.418 | (distorted: 23.941)
    	rel. l2 reconstruction error: 20.167 | (distorted: 7.058)
    	nn accuracy: 96.1 %
    	nn validation set accuracy: 84.6 %
    	nn verifier accuracy: 72.1 %
    
    
    
## Method: NN ALL
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_0.png)


    


     20%|██        |20.0/100 [01:07<04:29, 3.37s/epoch, accuracy=0.852, ideal=2.37, loss=1.5, psnr=21.8, |grad|=8.83]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_1.png)


    


     40%|████      |40.0/100 [02:14<03:21, 3.36s/epoch, accuracy=0.93, ideal=1.32, loss=1.14, psnr=22.4, |grad|=7.78] 

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_2.png)


    


     60%|██████    |60.0/100 [03:22<02:14, 3.36s/epoch, accuracy=0.953, ideal=1.54, loss=1.18, psnr=22.8, |grad|=8.8]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_3.png)


    


     80%|████████  |80.0/100 [04:29<01:07, 3.36s/epoch, accuracy=0.875, ideal=2.14, loss=1.2, psnr=23.2, |grad|=9.31] 

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_4.png)


    


    100%|██████████|100.0/100 [05:37<00:00, 3.36s/epoch, accuracy=0.945, ideal=1.26, loss=1.25, psnr=25.8, |grad|=10.6]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_5.png)


    


    100%|██████████|100.0/100 [05:37<00:00, 3.37s/epoch, accuracy=0.945, ideal=1.26, loss=1.25, psnr=25.8, |grad|=10.6]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_9.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_10.png)


    
    Results:
    	loss: 4.213
    	average PSNR: 24.563 | (distorted: 23.962)
    	rel. l2 reconstruction error: 10.973 | (distorted: 7.058)
    	nn accuracy: 96.1 %
    	nn validation set accuracy: 84.4 %
    	nn verifier accuracy: 73.0 %
    
    
    
## Method: NN ALL CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_0.png)


    


     20%|██        |20.0/100 [01:40<06:39, 4.99s/epoch, accuracy=0.938, ideal=3.02, loss=3.12, psnr=23.6, |grad|=3.05]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_1.png)


    


     40%|████      |40.0/100 [03:20<05:01, 5.02s/epoch, accuracy=0.93, ideal=3.13, loss=2.91, psnr=21.9, |grad|=4.93]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_2.png)


    


     60%|██████    |60.0/100 [05:00<03:19, 5.00s/epoch, accuracy=0.961, ideal=3.06, loss=3.01, psnr=24.5, |grad|=3.28]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_3.png)


    


     80%|████████  |80.0/100 [06:40<01:40, 5.01s/epoch, accuracy=0.977, ideal=3.06, loss=2.75, psnr=24.3, |grad|=3.02]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_4.png)


    


    100%|██████████|100.0/100 [08:20<00:00, 5.03s/epoch, accuracy=0.953, ideal=3.22, loss=2.76, psnr=24, |grad|=3.96] 

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_5.png)


    


    100%|██████████|100.0/100 [08:20<00:00, 5.01s/epoch, accuracy=0.953, ideal=3.22, loss=2.76, psnr=24, |grad|=3.96]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_9.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_CC_10.png)


    
    Results:
    	loss: 11.311
    	average PSNR: 23.547 | (distorted: 23.908)
    	rel. l2 reconstruction error: 18.713 | (distorted: 7.058)
    	nn accuracy: 95.1 %
    	nn validation set accuracy: 83.3 %
    	nn verifier accuracy: 69.1 %
    
    
    
## Method: RP
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_0.png)


    


     20%|██        |20.0/100 [01:03<04:15, 3.20s/epoch, accuracy=0.477, ideal=801, loss=645, psnr=16.7, |grad|=210]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_1.png)


    


     40%|████      |40.0/100 [02:07<03:11, 3.20s/epoch, accuracy=0.516, ideal=881, loss=734, psnr=18.2, |grad|=1.17e+3]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_2.png)


    


     60%|██████    |60.0/100 [03:11<02:07, 3.20s/epoch, accuracy=0.461, ideal=1.02e+3, loss=695, psnr=14.8, |grad|=340]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_3.png)


    


     80%|████████  |80.0/100 [04:15<01:03, 3.20s/epoch, accuracy=0.453, ideal=726, loss=661, psnr=14.2, |grad|=425]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_4.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.19s/epoch, accuracy=0.453, ideal=1.08e+3, loss=635, psnr=18.7, |grad|=425]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_5.png)


    


    100%|██████████|100.0/100 [05:20<00:00, 3.20s/epoch, accuracy=0.453, ideal=1.08e+3, loss=635, psnr=18.7, |grad|=425]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_8.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_9.png)


    
    Results:
    	loss: 2554.835
    	average PSNR: 17.473 | (distorted: 23.911)
    	rel. l2 reconstruction error: 49.880 | (distorted: 7.058)
    	nn accuracy: 44.5 %
    	nn validation set accuracy: 40.3 %
    	nn verifier accuracy: 30.9 %
    
    
    
## Method: RP CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_0.png)


    


     20%|██        |20.0/100 [01:05<04:21, 3.27s/epoch, accuracy=0.406, ideal=2.35e+3, loss=2.1e+3, psnr=18.3, |grad|=174] 

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_1.png)


    


     40%|████      |40.0/100 [02:10<03:15, 3.26s/epoch, accuracy=0.375, ideal=2.48e+3, loss=1.98e+3, psnr=14.6, |grad|=463]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_2.png)


    


     60%|██████    |60.0/100 [03:16<02:10, 3.27s/epoch, accuracy=0.367, ideal=2.42e+3, loss=2.04e+3, psnr=18.1, |grad|=417]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_3.png)


    


     80%|████████  |80.0/100 [04:21<01:05, 3.26s/epoch, accuracy=0.422, ideal=2.3e+3, loss=1.95e+3, psnr=16.4, |grad|=205]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_4.png)


    


    100%|██████████|100.0/100 [05:27<00:00, 3.27s/epoch, accuracy=0.367, ideal=2.57e+3, loss=2.06e+3, psnr=16.9, |grad|=222]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_5.png)


    


    100%|██████████|100.0/100 [05:27<00:00, 3.27s/epoch, accuracy=0.367, ideal=2.57e+3, loss=2.06e+3, psnr=16.9, |grad|=222]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_8.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_RP_CC_9.png)


    
    Results:
    	loss: 8066.062
    	average PSNR: 16.492 | (distorted: 23.872)
    	rel. l2 reconstruction error: 31.708 | (distorted: 7.058)
    	nn accuracy: 36.1 %
    	nn validation set accuracy: 33.9 %
    	nn verifier accuracy: 27.1 %
    
    
    
## Method: NN ALL + RP CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_0.png)


    


     20%|██        |20.0/100 [01:41<06:48, 5.11s/epoch, accuracy=0.906, ideal=68.9, loss=57.5, psnr=18.7, |grad|=14.6]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_1.png)


    


     40%|████      |40.0/100 [03:23<05:05, 5.10s/epoch, accuracy=0.898, ideal=64.2, loss=58, psnr=22.4, |grad|=8.51]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_2.png)


    


     60%|██████    |60.0/100 [05:05<03:23, 5.08s/epoch, accuracy=0.898, ideal=63.9, loss=60.2, psnr=21.9, |grad|=15.1]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_3.png)


    


     80%|████████  |80.0/100 [06:46<01:41, 5.06s/epoch, accuracy=0.922, ideal=68.2, loss=56.8, psnr=20.2, |grad|=7.5] 

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_4.png)


    


    100%|██████████|100.0/100 [08:28<00:00, 5.12s/epoch, accuracy=0.93, ideal=58.9, loss=53.7, psnr=22, |grad|=14.9]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_5.png)


    


    100%|██████████|100.0/100 [08:28<00:00, 5.09s/epoch, accuracy=0.93, ideal=58.9, loss=53.7, psnr=22, |grad|=14.9]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_9.png)


    Inverted:
    128 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_CIFAR10_NN_ALL_+_RP_CC_10.png)


    
    Results:
    	loss: 233.005
    	average PSNR: 21.296 | (distorted: 23.892)
    	rel. l2 reconstruction error: 8.410 | (distorted: 7.058)
    	nn accuracy: 91.2 %
    	nn validation set accuracy: 79.7 %
    	nn verifier accuracy: 65.0 %
    
# Summary
    =========
    
    
    baseline       acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    -----------------------------------------------------------
    B (original)   0.96  0.89      0.80      --        --      
    B (distorted)  0.56  0.51      0.44      23.89     7.06    
    A              0.97  --        0.79      --        --      
    
    Reconstruction methods:
    
    method          acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    ------------------------------------------------------------
    CRITERION       0.95  0.83      0.70      18.42     22.26   
    NN              0.89  0.76      0.63      16.92     16.28   
    NN CC           0.96  0.85      0.72      22.42     20.17   
    NN ALL          0.96  0.84      0.73      24.56     10.97   
    NN ALL CC       0.95  0.83      0.69      23.55     18.71   
    RP              0.45  0.40      0.31      17.47     49.88   
    RP CC           0.36  0.34      0.27      16.49     31.71   
    NN ALL + RP CC  0.91  0.80      0.65      21.30     8.41    


## MNIST



nbconv:
    images: 'reconstruction_mnist'


```
!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=MNIST \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=-1 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=0.001 \
-r_distort_level=0.5 \
--plot_ideal \
-show_after=20 \
```

    Already up to date.
    HEAD is now at b361314 print
# Testing reconstruction methods
# on MNIST
    Hyperparameters:
    dataset=MNIST
    seed=0
    nn_lr=0.01
    nn_steps=100
    batch_size=128
    n_random_projections=512
    inv_lr=0.1
    inv_steps=100
    f_reg=0.0
    f_crit=1.0
    f_stats=0.001
    size_A=-1
    size_B=512
    show_after=20
    r_distort_level=0.5
    r_block_depth=4
    r_block_width=4
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=False
    plot_ideal=True
    scale_each=False
    reset_stats=False 
    
    Running on 'cuda'
    
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet20.pt
    net accuracy: 90.6%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet9.pt
    verifier net accuracy: 96.2%
    
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-+-RP-CC-512.pt.
    
    ground truth:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_0.png)


    
    
    distorted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_1.png)


    
    
    
    
## Method: CRITERION
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_0.png)


    


     20%|██        |20.0/100 [00:24<01:39, 1.24s/epoch, accuracy=0.844, ideal=0.485, loss=0.685, psnr=12.9, |grad|=0.00649]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_1.png)


    


     40%|████      |40.0/100 [00:49<01:14, 1.24s/epoch, accuracy=0.875, ideal=0.131, loss=0.462, psnr=8.38, |grad|=0.0125]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_2.png)


    


     60%|██████    |60.0/100 [01:14<00:49, 1.25s/epoch, accuracy=0.867, ideal=0.285, loss=0.361, psnr=7.67, |grad|=0.0309]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_3.png)


    


     80%|████████  |80.0/100 [01:39<00:24, 1.25s/epoch, accuracy=0.898, ideal=0.272, loss=0.306, psnr=7.4, |grad|=0.023] 

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_4.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.24s/epoch, accuracy=0.773, ideal=0.428, loss=0.901, psnr=9.01, |grad|=0.0072]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_5.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.25s/epoch, accuracy=0.773, ideal=0.428, loss=0.901, psnr=9.01, |grad|=0.0072]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_7.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_CRITERION_8.png)


    
    Results:
    	loss: 3.610
    	average PSNR: 7.664 | (distorted: 6.970)
    	rel. l2 reconstruction error: 14.270 | (distorted: 24.183)
    	nn accuracy: 67.8 %
    	nn validation set accuracy: 65.3 %
    	nn verifier accuracy: 16.8 %
    
    
    
## Method: NN
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_0.png)


    


     20%|██        |20.0/100 [00:24<01:38, 1.24s/epoch, accuracy=0.133, ideal=5.85, loss=23.3, psnr=10.3, |grad|=0.0103] 

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_1.png)


    


     40%|████      |40.0/100 [00:49<01:16, 1.27s/epoch, accuracy=0.156, ideal=2.48, loss=11.3, psnr=8.42, |grad|=0.0309]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_2.png)


    


     60%|██████    |60.0/100 [01:14<00:50, 1.25s/epoch, accuracy=0.219, ideal=2.46, loss=9.14, psnr=8.76, |grad|=0.0436]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_3.png)


    


     80%|████████  |80.0/100 [01:39<00:25, 1.25s/epoch, accuracy=0.164, ideal=4.83, loss=8.19, psnr=8.01, |grad|=0.165]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_4.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.25s/epoch, accuracy=0.445, ideal=5.39, loss=6.31, psnr=10.4, |grad|=0.0701]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_5.png)


    


    100%|██████████|100.0/100 [02:04<00:00, 1.25s/epoch, accuracy=0.445, ideal=5.39, loss=6.31, psnr=10.4, |grad|=0.0701]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_8.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_9.png)


    
    Results:
    	loss: 27.875
    	average PSNR: 10.362 | (distorted: 6.970)
    	rel. l2 reconstruction error: 15.545 | (distorted: 24.183)
    	nn accuracy: 42.0 %
    	nn validation set accuracy: 41.2 %
    	nn verifier accuracy: 55.3 %
    
    
    
## Method: NN CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_0.png)


    


     20%|██        |20.0/100 [00:25<01:42, 1.29s/epoch, accuracy=0.438, ideal=2.28, loss=9.66, psnr=10.4, |grad|=0.0599]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_1.png)


    


     40%|████      |40.0/100 [00:51<01:18, 1.31s/epoch, accuracy=0.742, ideal=1.66, loss=6.16, psnr=12.4, |grad|=0.279]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_2.png)


    


     60%|██████    |60.0/100 [01:17<00:51, 1.29s/epoch, accuracy=0.172, ideal=2.24, loss=14.9, psnr=9.12, |grad|=0.048]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_3.png)


    


     80%|████████  |80.0/100 [01:43<00:25, 1.28s/epoch, accuracy=0.711, ideal=2.43, loss=5.67, psnr=10.6, |grad|=0.0058]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_4.png)


    


    100%|██████████|100.0/100 [02:09<00:00, 1.28s/epoch, accuracy=0.742, ideal=2.52, loss=4.36, psnr=10.5, |grad|=0.305]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_5.png)


    


    100%|██████████|100.0/100 [02:09<00:00, 1.30s/epoch, accuracy=0.742, ideal=2.52, loss=4.36, psnr=10.5, |grad|=0.305]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_8.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_CC_9.png)


    
    Results:
    	loss: 19.476
    	average PSNR: 10.392 | (distorted: 6.970)
    	rel. l2 reconstruction error: 13.238 | (distorted: 24.183)
    	nn accuracy: 75.0 %
    	nn validation set accuracy: 73.9 %
    	nn verifier accuracy: 52.9 %
    
    
    
## Method: NN ALL
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_0.png)


    


     20%|██        |20.0/100 [00:27<01:48, 1.36s/epoch, accuracy=0.844, ideal=1.09, loss=1.35, psnr=12.4, |grad|=0.0201]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_1.png)


    


     40%|████      |40.0/100 [00:54<01:21, 1.35s/epoch, accuracy=0.938, ideal=0.509, loss=0.741, psnr=12.3, |grad|=0.011]  

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_2.png)


    


     60%|██████    |60.0/100 [01:21<00:55, 1.38s/epoch, accuracy=0.93, ideal=0.585, loss=0.789, psnr=11.7, |grad|=0.00891]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_3.png)


    


     80%|████████  |80.0/100 [01:49<00:27, 1.36s/epoch, accuracy=0.961, ideal=0.774, loss=0.707, psnr=12.3, |grad|=0.0326]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_4.png)


    


    100%|██████████|100.0/100 [02:16<00:00, 1.36s/epoch, accuracy=0.891, ideal=1.07, loss=0.876, psnr=11.9, |grad|=0.0545]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_5.png)


    


    100%|██████████|100.0/100 [02:16<00:00, 1.37s/epoch, accuracy=0.891, ideal=1.07, loss=0.876, psnr=11.9, |grad|=0.0545]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_9.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_10.png)


    
    Results:
    	loss: 3.164
    	average PSNR: 11.573 | (distorted: 6.970)
    	rel. l2 reconstruction error: 12.511 | (distorted: 24.183)
    	nn accuracy: 93.0 %
    	nn validation set accuracy: 88.5 %
    	nn verifier accuracy: 87.1 %
    
    
    
## Method: NN ALL CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_0.png)


    


     20%|██        |20.0/100 [00:47<03:10, 2.38s/epoch, accuracy=0.789, ideal=0.924, loss=2.15, psnr=13.2, |grad|=0.0406]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_1.png)


    


     40%|████      |40.0/100 [01:35<02:22, 2.38s/epoch, accuracy=0.938, ideal=0.569, loss=1.21, psnr=11.7, |grad|=0.00674]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_2.png)


    


     60%|██████    |60.0/100 [02:23<01:35, 2.38s/epoch, accuracy=0.875, ideal=0.75, loss=1.04, psnr=11.5, |grad|=0.00996] 

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_3.png)


    


     80%|████████  |80.0/100 [03:11<00:47, 2.37s/epoch, accuracy=0.938, ideal=0.709, loss=0.885, psnr=11, |grad|=0.00259]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_4.png)


    


    100%|██████████|100.0/100 [03:58<00:00, 2.37s/epoch, accuracy=0.922, ideal=0.909, loss=1.07, psnr=11.2, |grad|=0.00539]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_5.png)


    


    100%|██████████|100.0/100 [03:58<00:00, 2.39s/epoch, accuracy=0.922, ideal=0.909, loss=1.07, psnr=11.2, |grad|=0.00539]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_9.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_CC_10.png)


    
    Results:
    	loss: 4.273
    	average PSNR: 11.324 | (distorted: 6.970)
    	rel. l2 reconstruction error: 15.535 | (distorted: 24.183)
    	nn accuracy: 91.0 %
    	nn validation set accuracy: 87.7 %
    	nn verifier accuracy: 91.0 %
    
    
    
## Method: RP
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_0.png)


    


     20%|██        |20.0/100 [00:25<01:40, 1.26s/epoch, accuracy=0.844, ideal=0.489, loss=0.703, psnr=13, |grad|=0.00325] 

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_1.png)


    


     40%|████      |40.0/100 [00:51<01:17, 1.29s/epoch, accuracy=0.82, ideal=0.135, loss=0.613, psnr=8.33, |grad|=0.0526] 

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_2.png)


    


     60%|██████    |60.0/100 [01:16<00:50, 1.26s/epoch, accuracy=0.844, ideal=0.289, loss=0.478, psnr=7.78, |grad|=0.00754]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_3.png)


    


     80%|████████  |80.0/100 [01:42<00:25, 1.27s/epoch, accuracy=0.906, ideal=0.275, loss=0.328, psnr=9.39, |grad|=0.0246]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_4.png)


    


    100%|██████████|100.0/100 [02:07<00:00, 1.27s/epoch, accuracy=0.898, ideal=0.432, loss=0.353, psnr=9.65, |grad|=0.0208]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_5.png)


    


    100%|██████████|100.0/100 [02:08<00:00, 1.28s/epoch, accuracy=0.898, ideal=0.432, loss=0.353, psnr=9.65, |grad|=0.0208]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_8.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_9.png)


    
    Results:
    	loss: 1.098
    	average PSNR: 9.227 | (distorted: 6.970)
    	rel. l2 reconstruction error: 15.509 | (distorted: 24.183)
    	nn accuracy: 92.8 %
    	nn validation set accuracy: 85.3 %
    	nn verifier accuracy: 24.8 %
    
    
    
## Method: RP CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_0.png)


    


     20%|██        |20.0/100 [00:27<01:49, 1.37s/epoch, accuracy=0.844, ideal=0.495, loss=0.706, psnr=13, |grad|=0.00247]  

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_1.png)


    


     40%|████      |40.0/100 [00:54<01:21, 1.35s/epoch, accuracy=0.906, ideal=0.141, loss=0.252, psnr=10.8, |grad|=0.00897]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_2.png)


    


     60%|██████    |60.0/100 [01:21<00:54, 1.35s/epoch, accuracy=0.93, ideal=0.295, loss=0.187, psnr=9.46, |grad|=0.00641]

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_3.png)


    


     80%|████████  |80.0/100 [01:48<00:26, 1.34s/epoch, accuracy=0.969, ideal=0.281, loss=0.123, psnr=9.93, |grad|=0.00253]

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_4.png)


    


    100%|██████████|100.0/100 [02:15<00:00, 1.34s/epoch, accuracy=0.961, ideal=0.439, loss=0.147, psnr=9.12, |grad|=0.00315]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_5.png)


    


    100%|██████████|100.0/100 [02:15<00:00, 1.36s/epoch, accuracy=0.961, ideal=0.439, loss=0.147, psnr=9.12, |grad|=0.00315]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_8.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_RP_CC_9.png)


    
    Results:
    	loss: 0.693
    	average PSNR: 9.155 | (distorted: 6.970)
    	rel. l2 reconstruction error: 14.112 | (distorted: 24.183)
    	nn accuracy: 96.3 %
    	nn validation set accuracy: 90.2 %
    	nn verifier accuracy: 62.9 %
    
    
    
## Method: NN ALL + RP CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_0.png)


    


     20%|██        |20.0/100 [00:50<03:22, 2.53s/epoch, accuracy=0.812, ideal=0.905, loss=2.01, psnr=13.7, |grad|=0.0162]

    
    epoch 20:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_1.png)


    


     40%|████      |40.0/100 [01:41<02:30, 2.51s/epoch, accuracy=0.914, ideal=0.55, loss=1.18, psnr=13.2, |grad|=0.000701]

    
    epoch 40:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_2.png)


    


     60%|██████    |60.0/100 [02:31<01:40, 2.50s/epoch, accuracy=0.875, ideal=0.73, loss=1.1, psnr=12.6, |grad|=0.00875]  

    
    epoch 60:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_3.png)


    


     80%|████████  |80.0/100 [03:22<00:51, 2.57s/epoch, accuracy=0.93, ideal=0.69, loss=0.906, psnr=11.9, |grad|=0.00261] 

    
    epoch 80:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_4.png)


    


    100%|██████████|100.0/100 [04:12<00:00, 2.52s/epoch, accuracy=0.906, ideal=0.888, loss=0.958, psnr=10.8, |grad|=0.0145]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_5.png)


    


    100%|██████████|100.0/100 [04:12<00:00, 2.53s/epoch, accuracy=0.906, ideal=0.888, loss=0.958, psnr=10.8, |grad|=0.0145]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_9.png)


    Inverted:
    50 / 512 



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/RECONSTRUCTION_MNIST_NN_ALL_+_RP_CC_10.png)


    
    Results:
    	loss: 3.857
    	average PSNR: 10.484 | (distorted: 6.970)
    	rel. l2 reconstruction error: 13.790 | (distorted: 24.183)
    	nn accuracy: 92.0 %
    	nn validation set accuracy: 87.2 %
    	nn verifier accuracy: 89.1 %
    
# Summary
    =========
    
    
    baseline       acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    -----------------------------------------------------------
    B (original)   0.89  0.90      0.96      --        --      
    B (distorted)  0.09  0.11      0.08      6.97      24.18   
    A              0.91  --        0.96      --        --      
    
    Reconstruction methods:
    
    method          acc   acc(val)  acc(ver)  av. PSNR  l2-err  
    ------------------------------------------------------------
    CRITERION       0.68  0.65      0.17      7.66      14.27   
    NN              0.42  0.41      0.55      10.36     15.55   
    NN CC           0.75  0.74      0.53      10.39     13.24   
    NN ALL          0.93  0.88      0.87      11.57     12.51   
    NN ALL CC       0.91  0.88      0.91      11.32     15.53   
    RP              0.93  0.85      0.25      9.23      15.51   
    RP CC           0.96  0.90      0.63      9.16      14.11   
    NN ALL + RP CC  0.92  0.87      0.89      10.48     13.79   



```
# deeper blocks
!git pull
!git reset --hard origin/master
%run reconstruction.py \
-dataset=MNIST \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=100 \
-size_A=-1 \
-size_B=512 \
-batch_size=128 \
-f_reg=0 \
-f_crit=1 \
-f_stats=0.001 \
-r_distort_level=0.5 \
-r_block_depth=8 \
--plot_ideal \
-show_after=20 \
--reset_stats \
```

# INVERSION

nbconv:
    images: 'inversion_cifar10'

## CIFAR10


```
# USING SCALE_EACH
!git pull
!git reset --hard origin/master
%run inversion.py \
-dataset=CIFAR10 \
-n_random_projections=512 \
-inv_lr=0.1 \
-inv_steps=700 \
-batch_size=256 \
-size_A=1024 \
-size_B=256 \
-f_reg=0.001 \
-f_crit=1 \
-f_stats=100 \
--use_jitter \
--plot_ideal \
--scale_each \

```

    Already up to date.
    HEAD is now at 8b1cf4e depth
# Testing reconstruction methods
# on CIFAR10
    Hyperparameters:
    dataset=CIFAR10
    seed=0
    nn_lr=0.01
    nn_steps=100
    batch_size=256
    n_random_projections=512
    inv_lr=0.1
    inv_steps=700
    f_reg=0.001
    f_crit=1.0
    f_stats=100.0
    size_A=1024
    size_B=256
    show_after=50
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=True
    plot_ideal=True
    scale_each=True 
    
    Running on 'cuda'
    
    Files already downloaded and verified
    Files already downloaded and verified
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet34.pt
    net accuracy: 96.5%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet50.pt
    verifier net accuracy: 81.8%
    
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-512.pt.


    100%|██████████| 4/4 [00:00<00:00, 18.27batch/s]

    
    Saving data to /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-ReLU-512.pt.


    
    100%|██████████| 4/4 [00:00<00:00, 16.85batch/s]

    
    Saving data to /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-ReLU-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-+-RP-CC-512.pt.
    
    
    
## Method: CRITERION
    
    
    epoch 0:


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_0.png)


    


      7%|▋         |50.0/700 [00:50<10:45,1.01epoch/s, accuracy=0.992, ideal=0.502, loss=0.568, |grad|=0.00687]

    
    epoch 50:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_1.png)


    


      9%|▉         |64.0/700 [01:04<10:30,1.01epoch/s, accuracy=1, ideal=0.502, loss=0.535, |grad|=0.00524]


```
# BASELINE, SANITY RUN

!git pull
!git reset --hard origin/master
%run ext/Nvlabs/cifar10/deepinversion-redo.py
```

    Already up to date.
    HEAD is now at 600a49c redo 700 iters
    loading resnet34
    Beginning Inversion.






    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_2.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_3.png)



```
# REDO MEAN + SCALE EACH
!git pull
!git reset --hard origin/master
%run inversion.py \
-dataset=CIFAR10 \
-inv_lr=0.1 \
-inv_steps=700 \
-batch_size=256 \
-size_B=256 \
-f_reg=0.001 \
-f_crit=1 \
-f_stats=10 \
-n_random_projections=256 \
-show_after=100
--use_jitter \
--plot_ideal \
--scale_each \

```

    Already up to date.
    HEAD is now at d8ceb64 stats comparability
# Testing reconstruction methods
# on CIFAR10
    Hyperparameters:
    dataset=CIFAR10
    seed=0
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=False
    plot_ideal=False
    nn_lr=0.01
    nn_steps=100
    batch_size=256
    n_random_projections=256
    inv_lr=0.1
    inv_steps=700
    f_reg=0.001
    f_crit=1.0
    f_stats=10.0
    size_A=-1
    size_B=256
    show_after=100 
    
    Running on 'cuda'
    
    Files already downloaded and verified
    Files already downloaded and verified
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet34.pt
    net accuracy: 96.2%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/CIFAR10/net_resnet50.pt
    verifier net accuracy: 80.6%
    
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/RP-256.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_RP-CC-256.pt.
    Loading data from /content/drive/My Drive/Thesis/models/CIFAR10/stats_NN-ALL-+-RP-CC-256.pt.
    
    
## Method: CRITERION
    


     14%|█▍        |101.0/700 [02:48<16:39,1.67s/epoch, accuracy=1, loss=0.581, |grad|=0.00437]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_0.png)


    


     29%|██▊       |201.0/700 [05:35<13:52,1.67s/epoch, accuracy=1, loss=0.385, |grad|=0.00545]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_1.png)


    


     43%|████▎     |301.0/700 [08:22<11:06,1.67s/epoch, accuracy=1, loss=0.374, |grad|=0.0067] 

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_2.png)


    


     57%|█████▋    |401.0/700 [11:09<08:18,1.67s/epoch, accuracy=1, loss=0.398, |grad|=0.00712]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_3.png)


    


     72%|███████▏  |501.0/700 [13:56<05:32,1.67s/epoch, accuracy=1, loss=0.376, |grad|=0.00629]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_4.png)


    


     86%|████████▌ |601.0/700 [16:43<02:45,1.67s/epoch, accuracy=1, loss=0.389, |grad|=0.00639]

    
    epoch 600:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_5.png)


    


    100%|██████████|700.0/700 [19:29<00:00,1.67s/epoch, accuracy=1, loss=0.378, |grad|=0.00628]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_7.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_CRITERION_8.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 39.8 %
    
    
## Method: NN
    


     14%|█▍        |101.0/700 [02:48<16:42,1.67s/epoch, accuracy=1, loss=0.712, |grad|=0.00392]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_0.png)


    


     29%|██▊       |201.0/700 [05:36<13:54,1.67s/epoch, accuracy=1, loss=0.404, |grad|=0.00486]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_1.png)


    


     43%|████▎     |301.0/700 [08:23<11:07,1.67s/epoch, accuracy=1, loss=0.395, |grad|=0.00506]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_2.png)


    


     57%|█████▋    |401.0/700 [11:11<08:20,1.67s/epoch, accuracy=1, loss=0.389, |grad|=0.00523]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_3.png)


    


     72%|███████▏  |501.0/700 [13:58<05:32,1.67s/epoch, accuracy=1, loss=0.416, |grad|=0.00469]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_4.png)


    


     86%|████████▌ |601.0/700 [16:46<02:45,1.67s/epoch, accuracy=1, loss=0.392, |grad|=0.00559]

    
    epoch 600:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_5.png)


    


    100%|██████████|700.0/700 [19:32<00:00,1.67s/epoch, accuracy=1, loss=0.4, |grad|=0.0115]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_8.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_9.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 41.0 %
    
    
## Method: NN CC
    


     14%|█▍        |101.0/700 [02:50<16:48,1.68s/epoch, accuracy=0.984, loss=1.17, |grad|=0.0916]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_0.png)


    


     29%|██▊       |201.0/700 [05:38<13:59,1.68s/epoch, accuracy=0.996, loss=1.11, |grad|=0.0611]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_1.png)


    


     43%|████▎     |301.0/700 [08:26<11:11,1.68s/epoch, accuracy=1, loss=0.953, |grad|=0.0137]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_2.png)


    


     57%|█████▋    |401.0/700 [11:15<08:22,1.68s/epoch, accuracy=0.996, loss=1.04, |grad|=0.042] 

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_3.png)


    


     72%|███████▏  |501.0/700 [14:03<05:34,1.68s/epoch, accuracy=1, loss=1.05, |grad|=0.0128]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_4.png)


    


     86%|████████▌ |601.0/700 [16:51<02:46,1.68s/epoch, accuracy=1, loss=0.936, |grad|=0.0161] 

    
    epoch 600:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_5.png)


    


    100%|██████████|700.0/700 [19:38<00:00,1.68s/epoch, accuracy=0.996, loss=1.72, |grad|=0.016]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_8.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_CC_9.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 64.1 %
    
    
## Method: NN ALL
    


     14%|█▍        |101.0/700 [02:53<17:11,1.72s/epoch, accuracy=0.992, loss=9.26, |grad|=0.0911]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_0.png)


    


     29%|██▊       |201.0/700 [05:46<14:19,1.72s/epoch, accuracy=0.992, loss=5.52, |grad|=0.0787]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_1.png)


    


     43%|████▎     |301.0/700 [08:38<11:27,1.72s/epoch, accuracy=0.996, loss=3.65, |grad|=0.0894]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_2.png)


    


     57%|█████▋    |401.0/700 [11:31<08:34,1.72s/epoch, accuracy=0.996, loss=2.82, |grad|=0.0825]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_3.png)


    


     72%|███████▏  |501.0/700 [14:23<05:42,1.72s/epoch, accuracy=0.996, loss=2.63, |grad|=0.0848]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_4.png)


    


     86%|████████▌ |601.0/700 [17:16<02:50,1.73s/epoch, accuracy=0.996, loss=2.46, |grad|=0.0807]

    
    epoch 600:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_5.png)


    


    100%|██████████|700.0/700 [20:07<00:00,1.72s/epoch, accuracy=1, loss=2.84, |grad|=0.0759]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_9.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_10.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 75.4 %
    
    
## Method: NN ALL CC
    


     14%|█▍        |101.0/700 [03:29<20:44,2.08s/epoch, accuracy=1, loss=1.57, |grad|=0.00304]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_0.png)


    


     29%|██▊       |201.0/700 [06:58<17:19,2.08s/epoch, accuracy=1, loss=1.26, |grad|=0.00259]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_1.png)


    


     43%|████▎     |301.0/700 [10:26<13:51,2.08s/epoch, accuracy=1, loss=1.01, |grad|=0.00239]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_2.png)


    


     57%|█████▋    |401.0/700 [13:54<10:21,2.08s/epoch, accuracy=1, loss=0.753, |grad|=0.00262]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_3.png)


    


     72%|███████▏  |501.0/700 [17:22<06:53,2.08s/epoch, accuracy=1, loss=0.697, |grad|=0.00326]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_4.png)


    


     86%|████████▌ |601.0/700 [20:50<03:25,2.08s/epoch, accuracy=1, loss=0.669, |grad|=0.00443]

    
    epoch 600:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_5.png)


    


    100%|██████████|700.0/700 [24:15<00:00,2.08s/epoch, accuracy=1, loss=0.653, |grad|=0.00409]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_9.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_NN_ALL_CC_10.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 41.8 %
    
    
## Method: RP
    


     14%|█▍        |101.0/700 [02:49<16:42,1.67s/epoch, accuracy=1, loss=4.28, |grad|=0.104]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_0.png)


    


     29%|██▊       |201.0/700 [05:36<13:54,1.67s/epoch, accuracy=1, loss=2.91, |grad|=0.105]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_1.png)


    


     43%|████▎     |301.0/700 [08:24<11:07,1.67s/epoch, accuracy=1, loss=2.53, |grad|=0.104]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_2.png)


    


     57%|█████▋    |401.0/700 [11:11<08:20,1.67s/epoch, accuracy=1, loss=2.35, |grad|=0.104]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_3.png)


    


     72%|███████▏  |501.0/700 [13:59<05:32,1.67s/epoch, accuracy=1, loss=2.31, |grad|=0.104]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_4.png)


    


     86%|████████▌ |601.0/700 [16:46<02:45,1.67s/epoch, accuracy=1, loss=2.14, |grad|=0.104]

    
    epoch 600:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_5.png)


    


    100%|██████████|700.0/700 [19:32<00:00,1.68s/epoch, accuracy=1, loss=2.19, |grad|=0.104]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_8.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_9.png)


    	nn accuracy: 100.0 %
    	nn verifier accuracy: 35.2 %
    
    
## Method: RP CC
    


     14%|█▍        |101.0/700 [02:50<16:52,1.69s/epoch, accuracy=0.801, loss=13.2, |grad|=2.12]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_CC_0.png)


    


     29%|██▊       |201.0/700 [05:39<14:03,1.69s/epoch, accuracy=0.891, loss=13.4, |grad|=2.13]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_CC_1.png)


    


     43%|████▎     |301.0/700 [08:29<11:14,1.69s/epoch, accuracy=0.934, loss=12.3, |grad|=2.04]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_CC_2.png)


    


     57%|█████▋    |401.0/700 [11:18<08:28,1.70s/epoch, accuracy=0.961, loss=13.2, |grad|=2.06]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_CC_3.png)


    


     72%|███████▏  |501.0/700 [14:07<05:36,1.69s/epoch, accuracy=0.973, loss=11.2, |grad|=2]   

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_CC_4.png)


    


     86%|████████▌ |601.0/700 [16:56<02:47,1.69s/epoch, accuracy=0.984, loss=10.3, |grad|=1.9] 

    
    epoch 600:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_CIFAR10_RP_CC_5.png)


    


     89%|████████▉ |625.0/700 [17:37<02:06,1.69s/epoch, accuracy=0.988, loss=9.98, |grad|=2.06]

nbconv:
    images: 'inversion_mnist'

## MNIST


```
!git pull
!git reset --hard origin/master
%run inversion.py \
-dataset=MNIST \
-n_random_projections=512 \
-nn_steps=1 \
-size_A=-1 \
-size_B=128 \
-batch_size=128 \
-inv_lr=0.05 \
-inv_steps=500 \
-f_reg=0.0005 \
-f_crit=1 \
-f_stats=0.001 \
-show_after=100 \
-seed=-1 \
--use_jitter \
--plot_ideal \

# --nn_resume_train \
# --reset_stats \

```

    Already up to date.
    HEAD is now at efb7fd3 inversion upd
# Testing reconstruction methods
# on MNIST
    Hyperparameters:
    dataset=MNIST
    seed=-1
    nn_lr=0.01
    nn_steps=1
    batch_size=128
    n_random_projections=512
    inv_lr=0.05
    inv_steps=500
    f_reg=0.0005
    f_crit=1.0
    f_stats=0.001
    size_A=-1
    size_B=128
    show_after=100
    nn_resume_train=False
    nn_reset_train=False
    use_amp=False
    use_std=False
    use_jitter=True
    plot_ideal=True
    scale_each=False
    reset_stats=False 
    
    Running on 'cuda'
    
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet20.pt
    net accuracy: 99.4%
    Training Checkpoint restored: /content/drive/My Drive/Thesis/models/MNIST/net_resnet9.pt
    verifier net accuracy: 95.9%
    
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/RP-512.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_inputs-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-CC.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_RP-CC-512.pt.
    Loading data from /content/drive/My Drive/Thesis/models/MNIST/stats_NN-ALL-+-RP-CC-512.pt.
    
    
    
## Method: CRITERION
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_0.png)


    


     20%|██        |100.0/500 [00:15<01:00,6.65epoch/s, accuracy=0.961, ideal=0.058, loss=0.208, |grad|=0.00124]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_1.png)


    


     40%|████      |200.0/500 [00:30<00:45,6.63epoch/s, accuracy=0.945, ideal=0.058, loss=0.34, |grad|=0.00205]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_2.png)


    


     60%|██████    |300.0/500 [00:45<00:30,6.66epoch/s, accuracy=0.93, ideal=0.058, loss=0.685, |grad|=0.00274]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_3.png)


    


     80%|████████  |400.0/500 [01:00<00:15,6.61epoch/s, accuracy=0.922, ideal=0.058, loss=1.37, |grad|=0.00335]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_4.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.73epoch/s, accuracy=0.922, ideal=0.058, loss=1.5, |grad|=0.00273] 

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_5.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.60epoch/s, accuracy=0.922, ideal=0.058, loss=1.5, |grad|=0.00273]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_7.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_CRITERION_8.png)


    
    	nn accuracy: 92.2 %
    	nn verifier accuracy: 21.9 %
    
    
    
## Method: NN
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_0.png)


    


     20%|██        |100.0/500 [00:15<01:00,6.59epoch/s, accuracy=0.938, ideal=31.3, loss=4.22, |grad|=0.0164]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_1.png)


    


     40%|████      |200.0/500 [00:30<00:44,6.68epoch/s, accuracy=0.945, ideal=31.3, loss=3.87, |grad|=0.0075]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_2.png)


    


     60%|██████    |300.0/500 [00:45<00:29,6.69epoch/s, accuracy=0.945, ideal=31.3, loss=4.02, |grad|=0.00296]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_3.png)


    


     80%|████████  |400.0/500 [01:00<00:14,6.76epoch/s, accuracy=0.891, ideal=31.3, loss=4.48, |grad|=0.0185]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_4.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.66epoch/s, accuracy=0.875, ideal=31.3, loss=5.72, |grad|=0.0273]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_5.png)


    


    100%|██████████|500.0/500 [01:15<00:00,6.63epoch/s, accuracy=0.875, ideal=31.3, loss=5.72, |grad|=0.0273]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_8.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_9.png)


    
    	nn accuracy: 89.8 %
    	nn verifier accuracy: 18.8 %
    
    
    
## Method: NN CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_0.png)


    


     20%|██        |100.0/500 [00:15<01:02,6.36epoch/s, accuracy=0.969, ideal=22.7, loss=4.41, |grad|=0.0331]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_1.png)


    


     40%|████      |200.0/500 [00:31<00:47,6.36epoch/s, accuracy=0.906, ideal=22.7, loss=4.02, |grad|=0.0121]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_2.png)


    


     60%|██████    |300.0/500 [00:47<00:31,6.33epoch/s, accuracy=0.961, ideal=22.7, loss=3.76, |grad|=0.0122]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_3.png)


    


     80%|████████  |400.0/500 [01:03<00:15,6.34epoch/s, accuracy=0.953, ideal=22.7, loss=4.12, |grad|=0.0236]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_4.png)


    


    100%|██████████|500.0/500 [01:19<00:00,6.26epoch/s, accuracy=0.938, ideal=22.7, loss=4.33, |grad|=0.0207]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_5.png)


    


    100%|██████████|500.0/500 [01:19<00:00,6.27epoch/s, accuracy=0.938, ideal=22.7, loss=4.33, |grad|=0.0207]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_8.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_CC_9.png)


    
    	nn accuracy: 93.0 %
    	nn verifier accuracy: 18.0 %
    
    
    
## Method: NN ALL
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_0.png)


    


     20%|██        |100.0/500 [00:16<01:07,5.94epoch/s, accuracy=0.938, ideal=4.63, loss=1.86, |grad|=0.00254]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_1.png)


    


     40%|████      |200.0/500 [00:33<00:50,5.92epoch/s, accuracy=0.992, ideal=4.63, loss=1.45, |grad|=0.00173]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_2.png)


    


     60%|██████    |300.0/500 [00:50<00:33,5.98epoch/s, accuracy=0.977, ideal=4.63, loss=1.44, |grad|=0.00135]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_3.png)


    


     80%|████████  |400.0/500 [01:07<00:16,5.89epoch/s, accuracy=0.977, ideal=4.63, loss=1.43, |grad|=0.00157]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_4.png)


    


    100%|██████████|500.0/500 [01:24<00:00,5.95epoch/s, accuracy=0.969, ideal=4.63, loss=1.49, |grad|=0.00403]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_5.png)


    


    100%|██████████|500.0/500 [01:24<00:00,5.89epoch/s, accuracy=0.969, ideal=4.63, loss=1.49, |grad|=0.00403]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_9.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_10.png)


    
    	nn accuracy: 98.4 %
    	nn verifier accuracy: 10.2 %
    
    
    
## Method: NN ALL CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_0.png)


    


     20%|██        |100.0/500 [00:34<02:18,2.88epoch/s, accuracy=0.977, ideal=4.21, loss=1.99, |grad|=0.00301]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_1.png)


    


     40%|████      |200.0/500 [01:09<01:45,2.85epoch/s, accuracy=0.984, ideal=4.21, loss=1.7, |grad|=0.00178] 

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_2.png)


    


     60%|██████    |300.0/500 [01:43<01:09,2.90epoch/s, accuracy=0.984, ideal=4.21, loss=1.62, |grad|=0.00185]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_3.png)


    


     80%|████████  |400.0/500 [02:18<00:34,2.87epoch/s, accuracy=1, ideal=4.21, loss=1.48, |grad|=0.00231]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_4.png)


    


    100%|██████████|500.0/500 [02:53<00:00,2.85epoch/s, accuracy=0.984, ideal=4.21, loss=1.47, |grad|=0.00171]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_5.png)


    


    100%|██████████|500.0/500 [02:53<00:00,2.88epoch/s, accuracy=0.984, ideal=4.21, loss=1.47, |grad|=0.00171]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_9.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_CC_10.png)


    
    	nn accuracy: 100.0 %
    	nn verifier accuracy: 14.8 %
    
    
    
## Method: RP
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_0.png)


    


     20%|██        |100.0/500 [00:14<00:59,6.72epoch/s, accuracy=0.953, ideal=0.0616, loss=0.231, |grad|=0.00186]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_1.png)


    


     40%|████      |200.0/500 [00:29<00:44,6.67epoch/s, accuracy=0.953, ideal=0.0616, loss=0.447, |grad|=0.00229]

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_2.png)


    


     60%|██████    |300.0/500 [00:45<00:29,6.75epoch/s, accuracy=0.938, ideal=0.0616, loss=0.746, |grad|=0.00239]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_3.png)


    


     80%|████████  |400.0/500 [00:59<00:14,6.79epoch/s, accuracy=0.93, ideal=0.0616, loss=1.02, |grad|=0.00219] 

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_4.png)


    


    100%|██████████|500.0/500 [01:14<00:00,6.78epoch/s, accuracy=0.922, ideal=0.0616, loss=1.48, |grad|=0.00268]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_5.png)


    


    100%|██████████|500.0/500 [01:14<00:00,6.67epoch/s, accuracy=0.922, ideal=0.0616, loss=1.48, |grad|=0.00268]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_8.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_9.png)


    
    	nn accuracy: 91.4 %
    	nn verifier accuracy: 23.4 %
    
    
    
## Method: RP CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_0.png)


    


     20%|██        |100.0/500 [00:16<01:06,6.06epoch/s, accuracy=0.953, ideal=0.0679, loss=0.221, |grad|=0.00142]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_1.png)


    


     40%|████      |200.0/500 [00:33<00:49,6.10epoch/s, accuracy=0.961, ideal=0.0679, loss=0.289, |grad|=0.002]  

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_2.png)


    


     60%|██████    |300.0/500 [00:49<00:32,6.07epoch/s, accuracy=0.938, ideal=0.0679, loss=0.682, |grad|=0.00248]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_3.png)


    


     80%|████████  |400.0/500 [01:06<00:16,6.03epoch/s, accuracy=0.922, ideal=0.0679, loss=1.07, |grad|=0.00269]

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_4.png)


    


    100%|██████████|500.0/500 [01:23<00:00,6.11epoch/s, accuracy=0.914, ideal=0.0679, loss=1.65, |grad|=0.00284]

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_5.png)


    


    100%|██████████|500.0/500 [01:23<00:00,6.00epoch/s, accuracy=0.914, ideal=0.0679, loss=1.65, |grad|=0.00284]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_8.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_RP_CC_9.png)


    
    	nn accuracy: 92.2 %
    	nn verifier accuracy: 23.4 %
    
    
    
## Method: NN ALL + RP CC
    
    
    epoch 0:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_0.png)


    


     20%|██        |100.0/500 [00:36<02:25,2.76epoch/s, accuracy=0.953, ideal=4.03, loss=1.91, |grad|=0.00159]

    
    epoch 100:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_1.png)


    


     40%|████      |200.0/500 [01:12<01:49,2.74epoch/s, accuracy=1, ideal=4.03, loss=1.6, |grad|=0.00114] 

    
    epoch 200:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_2.png)


    


     60%|██████    |300.0/500 [01:48<01:12,2.77epoch/s, accuracy=0.992, ideal=4.03, loss=1.52, |grad|=0.00152]

    
    epoch 300:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_3.png)


    


     80%|████████  |400.0/500 [02:25<00:37,2.66epoch/s, accuracy=0.992, ideal=4.03, loss=1.46, |grad|=0.0019] 

    
    epoch 400:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_4.png)


    


    100%|██████████|500.0/500 [03:01<00:00,2.76epoch/s, accuracy=1, ideal=4.03, loss=1.39, |grad|=0.00178]   

    
    epoch 500:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_5.png)


    


    100%|██████████|500.0/500 [03:01<00:00,2.75epoch/s, accuracy=1, ideal=4.03, loss=1.39, |grad|=0.00178]

    


    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_6.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_7.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_8.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_9.png)


    Inverted:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/INVERSION_MNIST_NN_ALL_+_RP_CC_10.png)


    
    	nn accuracy: 99.2 %
    	nn verifier accuracy: 18.8 %
    
# Summary
    =========
    
    
    method          acc   acc(ver)  
    --------------------------------
    CRITERION       0.92  0.22      
    NN              0.90  0.19      
    NN CC           0.93  0.18      
    NN ALL          0.98  0.10      
    NN ALL CC       1.00  0.15      
    RP              0.91  0.23      
    RP CC           0.92  0.23      
    NN ALL + RP CC  0.99  0.19      


nbconv:
    out: 'Tests'

# ->END
