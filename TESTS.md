
## Test 1


```
python projectOK/test1.py
```

    Simple regression test for recovering an affine transformation..
    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_10_1.png)


    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_10_5.png)


    After Pre-Processing:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_10_7.png)


    l2 reconstruction error: 0.886


## Test 2


```
python projectOK/test2.py
```

    ERROR:root:File `'projectOK/test2.py'` not found.


## Test 3


```
python projectOK/test3.py
```

    Testing reconstruction by matching statistics
    of Gaussian Mixture Model on random projections..
    
    Before:
    Cross Entropy of A: 3.711257219314575
    Cross Entropy of B: 1088.7333984375


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_2.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_3.png)


    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_7.png)


    After Pre-Processing:
    Cross Entropy of B: 4.431112289428711
    Cross Entropy of undistorted B: 3.6967499256134033



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_9.png)


    l2 reconstruction error: 7.680



```
python projectOK/test3bigger.py
```

    Test3 using random projections
    on 2 distinct clusters
    Comment:
    Method struggles with more seperated clusters
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_2.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_3.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_7.png)


    After Pre-Processing:
    Cross Entropy of B: inf
    Cross Entropy of undistorted B: 4.961302757263184



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_9.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_10.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_11.png)


    l2 reconstruction error: 35.574


## Test 4


```
python projectOK/test4.py
```

    Testing reconstruction by matching
    class-conditional statistics
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_17_2.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_17_6.png)


    After Pre-Processing:
    Cross Entropy of B: 6.816658973693848
    Cross Entropy of undistorted B: 4.961302757263184



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_17_8.png)


    l2 reconstruction error: 1.083


## Test 5


```
python projectOK/test5.py
```

    Testing reconstruction by matching
    class-conditional statistics on random projections
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_19_2.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_19_3.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_19_7.png)


    After Pre-Processing:
    Cross Entropy of B: 6.226280689239502
    Cross Entropy of undistorted B: 4.961302757263184



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_19_9.png)


    l2 reconstruction error: 1.945


## Test 6


```
python projectOK/test6.py
```

    Testing reconstruction by matching
    statistics on neural network feature
    
    Training Checkpoint restored: /content/Thesis/models/net_GMM_2-16-16-16-3.pt


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_21_2.png)


    Before:
    Cross Entropy of A: 5.5779805183410645
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_21_6.png)


    After Pre-Processing:
    Cross Entropy of B: 6.1775803565979



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_21_8.png)

