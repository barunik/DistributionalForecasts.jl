
# DistributionalForecasts.jl

[![GitHub](http://pkg.julialang.org/badges/GitHub_0.6.svg)](http://pkg.julialang.org/detail/GitHub)

The code has been developed in Julia 0.6.4. version, as a code accompanying the Anatolyev and Barunik (20xx) paper, and provides an estimation and inference for a model forecasting conditional probability distributions of asset returns (henceforth AB model). For further details, see

Anatolyev, S. and Barunik, J. (20xx): *Forecasting dynamic return distributions based on ordered binary choice and  cross-quantile predictability connection*, manuscript [available here for download](https://ideas.repec.org/p/arx/papers/1711.05681.html)


## Software requirements

[Julia](http://julialang.org/) together with few packages needs to be installed

````julia
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("GLM")
Pkg.add("Optim")
````

## Example: Forecasting dynamic return distributions

Note the full example is available as an interactive [IJulia](https://github.com/JuliaLang/IJulia.jl) notebook [here](https://github.com/barunik/DistributionalForecasts.jl/blob/master/Example_online.ipynb)


Load required packages


```julia
using DataFrames, CSV, GLM, Optim 

# load main functions
include("main.jl");
```

Load example data (returns of XOM)


```julia
data = CSV.read("data_30stocks_returns.txt");
tdim, rdim = size(data)
```

Choose number of cutoff levels and order of polynomials


```julia
# no. of quantiles
js = 37;

# choice of polynomial order
p1=2;
p2=3;
```

## Parameter Estimation

Obtain fast parameter estimates of AB without inference. A vector of $js+p1+p2+2$ parameters is returned:

<img src="https://latex.codecogs.com/gif.latex? $$\delta_{0,1},\delta_{0,2},...,\delta_{0,js},\kappa_{0,1},...\kappa_{p1+1,1},\kappa_{0,2},...\kappa_{p2+1,2}$$" /> 


```julia
par=OrderedLogitparameters(data[:,30].*1.0,js,p1,p2)
par'
```




    1×44 RowVector{Float64,Array{Float64,1}}:
     -2.87124  -2.48832  -2.2352  -2.0323  …  -17.5198  -16.8475  25.976



## Inference

Estimate the AB model and obtain full inference and evaluation of fit


```julia
est=OrderedLogit(data[:,30].*1.0, js,p1,p2);
```

Estimates of intercepts $\delta_{0,1},\delta_{0,2},...,\delta_{0,js}$


```julia
est[1][1:js]'
```




    1×37 RowVector{Float64,Array{Float64,1}}:
     -2.87124  -2.48832  -2.2352  -2.0323  …  2.10511  2.47381  2.57944  3.04193



Estimates of $\kappa_{0,1},...\kappa_{p1+1,1},\kappa_{0,2},...\kappa_{p2+1,2}$


```julia
est[1][(js+1):(js+p1+p2+2)]'
```




    1×7 RowVector{Float64,Array{Float64,1}}:
     -0.0528382  -0.116755  0.0523652  0.108634  -17.5198  -16.8475  25.976



standard errors for all coefficients


```julia
est[2]'
```




    1×44 RowVector{Float64,Array{Float64,1}}:
     0.117504  0.101611  0.0915005  …  8.10445  5.70739  7.8312  9.23371



T-stats


```julia
est[3]'
```




    1×44 RowVector{Float64,Array{Float64,1}}:
     -24.4352  -24.4886  -24.4283  -23.1584  …  -3.06967  -2.15133  2.81317



Log Likelihood


```julia
est[4]
```




    10285.797921780777



Information criteria (AIC/BIC)


```julia
est[5:6]
```




    2-element Array{Any,1}:
     -20482.2
     -20222.0



## Recover Probabilities Predicted by the AB model

Obtain forecast of return distribution for time $t+1$ based on the in-sample window


```julia
window=500
INS=data[1:window,30].*1
OOS=data[window:(window+1),30].*1

probs=forecastProbs(INS,OOS,js,p1,p2)
```




    1×37 Array{Float64,2}:
     0.0670917  0.0868857  0.108024  0.124577  …  0.928052  0.928632  0.951298



## Statistical Evaluation

A number of statistical tests are implemented in the *main.jl* file. TBD
