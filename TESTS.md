```
python projectOK/test1.py
```

    Simple regression test for recovering an affine transformation..
    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_9_1.png)


    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_9_5.png)


    After Pre-Processing:



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_9_7.png)


    l2 reconstruction error: 0.886



```
python projectOK/test2.py
```

    Testing reconstruction by matching statistics
    of Gaussian Mixture Model in input space..
    
    Comment:
    Statistics are matched, but data is deformed.
    Not enough information given.
    
    Before:


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_10_2.png)


    Cross Entropy of A: 4.47373104095459
    Cross Entropy of B: 10.691061973571777
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_10_6.png)


    After Pre-Processing:
    Cross Entropy of B: 5.985135078430176
    Cross Entropy of unperturbed B: 4.654342174530029



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_10_8.png)


    l2 reconstruction error: 3.823



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



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_11_2.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_11_3.png)


    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_11_7.png)


    After Pre-Processing:
    Cross Entropy of B: 4.431112289428711
    Cross Entropy of unperturbed B: 3.6967499256134033



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_11_9.png)


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



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_12_2.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_12_3.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_12_7.png)


    After Pre-Processing:
    Cross Entropy of B: inf
    Cross Entropy of unperturbed B: 4.961302757263184



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_12_9.png)


    l2 reconstruction error: 35.574



```
python projectOK/test4.py
```

    Testing reconstruction by matching
    class-conditional statistics
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_13_2.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_13_6.png)


    After Pre-Processing:
    Cross Entropy of B: 6.816658973693848
    Cross Entropy of unperturbed B: 4.961302757263184



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_13_8.png)


    l2 reconstruction error: 1.083



```
python projectOK/test5.py
```

    Testing reconstruction by matching
    class-conditional statistics on random projections
    


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_2.png)



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_3.png)


    Before:
    Cross Entropy of A: 5.0765461921691895
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_7.png)


    After Pre-Processing:
    Cross Entropy of B: 6.226280689239502
    Cross Entropy of unperturbed B: 4.961302757263184



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_14_9.png)


    l2 reconstruction error: 1.945



```
python projectOK/test6.py
```

    Testing reconstruction by matching
    statistics on neural network feature
    
    Training Checkpoint restored: /content/Thesis/models/net_GMM_2-16-16-16-3.pt


    /usr/local/lib/python3.6/dist-packages/scipy/stats/_multivariate.py:660: RuntimeWarning: covariance is not positive-semidefinite.
      out = random_state.multivariate_normal(mean, cov, size)
    /content/Thesis/utility.py:503: UserWarning: linewidths is ignored by contourf
      alpha=alpha, linewidths=5, linestyles='solid')



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_2.png)


    Before:
    Cross Entropy of A: 5.5779805183410645
    Cross Entropy of B: inf
    Beginning Inversion.





    



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_6.png)


    After Pre-Processing:
    Cross Entropy of B: 6.1775803565979



![png](https://raw.githubusercontent.com/willisk/Thesis/master/figures/README/README_15_8.png)

