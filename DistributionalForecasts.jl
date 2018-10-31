
## Exponential moving average
function EWMA(x)
    ewmaVar = zeros(length(x),1)
    ewmaVar[1] = var(x)
    for i in 2:length(x)
         ewmaVar[i] = 0.94*ewmaVar[i-1]+ (1.0 - 0.94)*x[i-1]^2.0
    end
    ewmaVol = sqrt.(ewmaVar);
    return ewmaVol
end

## OLS estimate
function OLSestimator(y,x)
    estimate = (transpose(x)*x) \ (transpose(x)*y)
    return estimate
end

## Function to generate polynoms
function delta3j(js)
    alphaj =  collect(linspace(0.05, 0.95, js))
    delta0 = collect(linspace(1.0, 1.0, length(alphaj)))
    delta1 = 2.0*(alphaj - 0.5)
    delta2 = (2.0^2.0)*((alphaj - 0.5).^2.0)
    delta3 = (2.0^3.0)*((alphaj - 0.5).^3.0)
    delta4 = (2.0^4.0)*((alphaj - 0.5).^4.0)
    delta5 = (2.0^5.0)*((alphaj - 0.5).^5.0)
    return [delta0 delta1 delta2 delta3 delta4 delta5]
end

## Fucntion to obtain initial estimates
function getinitlogit(rets, js,p1,p2)
    n = length(rets)
    x2 = log.(1 + abs.(rets))[1:(n-1)];  
    alphaj = collect(linspace(0.05, 0.95, js))
    coefdf = zeros(js, 3)
    estlog = 0.0
    for j in collect(1:(js))
        indx1 = (rets .< (EWMA(rets).*quantile.(Normal(0,1), alphaj[j])) ).*1.0;
        data1 = DataFrame([indx1[2:n], indx1[1:(n-1)], x2], [:y, :x1, :x2]);

        estlog = glm(@formula(y ~ x1 + x2), data1, Binomial(), LogitLink());
        coefdf[j,:] = coef(estlog)
    end

    ## Estimation of kappas from deltas 
    delta = delta3j(js)
    kapa1 = OLSestimator(coefdf[:, 2], delta[:,1:(p1+1)]);
    kapa2 = OLSestimator(coefdf[:, 3], delta[:,1:(p2+1)]);
    kapainit = [coefdf[:, 1]; kapa1; kapa2]
    return kapainit
end


## LL function             
function loglikeFinalstd(x, param0,p1,p2)
    ##
    rets = x
    n = length(rets)
    m = zeros(n)
    theta = zeros(n)
    n_alpha = length(param0) - (p1+p2+2) 
    prob = zeros(n, n_alpha)

    alphaj = collect(linspace(0.05, 0.95, n_alpha))

    cuts=zeros(n,n_alpha);
    for i in collect(1:n_alpha)
        cuts[:,i]=EWMA(rets)*quantile.(Normal(0,1), alphaj[i])
    end
    ##
    powers = [p1; p2]
    ## looping over time=j, and quantiles=i
    for j in collect(1:n)
        for i in collect(1:n_alpha)
            ##
            theta[j] = param0[1 + (i-1)*1]
            poly_1 = param0[1 + n_alpha]
            poly_2 = param0[2 + n_alpha + powers[1]]
            ## first polynoms in delta1
            for l in collect(1:powers[1])
                poly_1 = poly_1 + param0[1 + n_alpha + l] * (2.0 * alphaj[i] - 1.0)^l
            end
            ## second polynoms in delta2
            for l in collect(1:powers[2])
                poly_2 = poly_2 + param0[2 + n_alpha + powers[1]+l] * (2.0 * alphaj[i] - 1.0)^l
            end
            ##
            if j==1
                theta[j]=theta[j] + poly_1*mean(rets.<cuts[:,i]) + poly_2*mean(log.(1.0+abs.(rets)))
            else
                theta[j]=theta[j] + poly_1*(rets[j-1]<cuts[j-1,i]) + poly_2*log.(1.0+abs.(rets[j-1]))
            end
            ##
            prob[j, i] = 1.0 / (1.0 + exp.(-theta[j]))
        end
        m[j] = (rets[j] <= cuts[j,1]) * log.(prob[j, 1])
        for i in collect(2:n_alpha)
            if (prob[j, i] - prob[j, i-1]) <= 0.0
                prob[j, i] = prob[j, i-1] + 0.00001
            end
        
            m[j] = m[j] + ((rets[j] <= cuts[j,i]) * (rets[j] > cuts[j,i-1])) * log.(prob[j, i] - prob[j, i-1])
        end
            if (1.0 - prob[j, n_alpha]) <= 0.0
                prob[j, n_alpha] = 0.99999
            end
        m[j] = m[j] + (rets[j] > cuts[j,n_alpha]) * log.(1.0 - prob[j, n_alpha])
    end
    ##
    return m
end

function loglikeFinal(x, param0,p1,p2)
    
    m = loglikeFinalstd(x, param0,p1,p2)
    
    return -sum(m)
end

# Estimate parameters
function OrderedLogitparameters(x,q,p1,p2)

    kapainit = getinitlogit(x,q,p1,p2) 
    estone = optimize(kapainit -> loglikeFinal(x, kapainit,p1,p2), kapainit, NelderMead(), Optim.Options(g_tol=1e-5, iterations=100))

    par = estone.minimizer

    return par
end    

# Parameters and inference
function OrderedLogit(x,q,p1,p2)

    kapainit = getinitlogit(x,q,p1,p2) 
    estone = optimize(kapainit -> loglikeFinal(x, kapainit,p1,p2), kapainit, NelderMead(), Optim.Options(g_tol=1e-5, iterations=100))

    par = estone.minimizer

    n=length(x)
    b=par'
    k=length(b)
    frac = 0.00001;
    h=frac*b

    e = eye(k)

    DfDp=zeros(n,k)
    for i in 1:k
        DfDp[:,i] = (loglikeFinalstd(x, b'+h'.*e[i,:],p1,p2)-loglikeFinalstd(x, b'-h'.*e[i,:],p1,p2))./h[i]/2.0
    end

    DfDp2=zeros(k,k);
    for i in 1:k
        for j in 1:i
            DfDp2[i,j] = mean(DfDp[:,i].*DfDp[:,j])
            DfDp2[j,i] = DfDp2[i,j];
        end                                                                                                      
    end     

    Df2Dp=zeros(k,k);
    for i in 1:k
    for j in 1:i
        if i==j; 
            Df2Dp[i,i] = mean((loglikeFinalstd(x, b'+h'.*e[i,:],p1,p2)-2.0*loglikeFinalstd(x, b,p1,p2)+loglikeFinalstd(x, b'-h'.*e[i,:],p1,p2))./(h[i]^2.0));
        else 
            Df2Dp[i,j] = mean((loglikeFinalstd(x, b'+h'.*(e[i,:]+e[j,:]),p1,p2)-loglikeFinalstd(x, b'+h'.*(e[i,:]-e[j,:]),p1,p2)-loglikeFinalstd(x, b'+h'.*(e[j,:]-e[i,:]),p1,p2)+loglikeFinalstd(x, b'-h'.*(e[i,:]+e[j,:]),p1,p2))./(4.0*h[i]*h[j])); 
            Df2Dp[j,i] = Df2Dp[i,j];
        end
    end                                                                                                        
    end                                                                                                        
    Df2Dp=-Df2Dp;
    J = inv(Df2Dp)

    sterr=sqrt.(diag(J*DfDp2*J)/n)
    t=b'./sterr
    LL=estone.minimum

    AIC = -2LL + 2k + 2k*(k+1)/(n-k-1)
    BIC = -2LL + k*log(n)

    return [par,sterr,t,LL,AIC,BIC]
end                   



function forecastProbs(x, y, q,p1,p2)

    kapainit = getinitlogit(x,q,p1,p2) 

    estone = optimize(kapainit -> loglikeFinal(x, kapainit,p1,p2), kapainit, NelderMead(), Optim.Options(g_tol=1e-5, iterations=100))
    param0 = estone.minimizer

    rets0 = x
    rets = y
    n = length(rets)
    m = zeros(n)
    theta = zeros(n)
    n_alpha = length(param0) - (p1+p2+2)
    prob = zeros(n, n_alpha)
    alphaj = collect(linspace(0.05, 0.95, n_alpha))
    cuts=EWMA(rets)[length(rets)].*quantile.(Normal(0,1), alphaj)
    powers = [p1; p2]
    wrong=zeros(n)
    ## looping over time=j, and quantiles=i
    for j in collect(2:n)
        for i in collect(1:n_alpha)
            ##
            theta[j] = param0[1 + (i-1)*1]
            poly_1 = param0[1 + n_alpha]
            poly_2 = param0[2 + n_alpha + powers[1]]
            ## first polynoms in delta1
            for l in collect(1:powers[1])
                poly_1 = poly_1 + param0[1 + n_alpha + l] * (2.0 * alphaj[i] - 1.0)^l
            end
            ## second polynoms in delta2
            for l in collect(1:powers[2])
                poly_2 = poly_2 + param0[2 + n_alpha + powers[1]+l] * (2.0 * alphaj[i] - 1.0)^l
            end

            theta[j]=theta[j] + poly_1*(rets[j-1]<cuts[i]) + poly_2*log.(1.0+abs.(rets[j-1]))

            prob[j, i] = 1.0 / (1.0 + exp.(-theta[j]))
        end
        for i in collect(2:n_alpha)
            if (prob[j, i] - prob[j, i-1]) <= 0.0
                    prob[j, i] = prob[j, i-1] + 0.00001
                    #wrong[j] = 1
            end
        end
    end

    return prob[2:n,:]
end

## ------------------------------------------
## Statistical evaluation of forecasts
## ------------------------------------------


function Interpol(xs,ys,x)
    n = length(xs)
    dxs = xs[2:n]-xs[1:n-1]
    dys = ys[2:n]-ys[1:n-1] 
    Ds = dys./dxs
    ms = [Ds[1];(Ds[1:n-2]+Ds[2:n-1])/2;Ds[n-1]]
    as = ms[1:n-1]./Ds[1:n-1] 
    bs = ms[2:n]./Ds[1:n-1]
    cir = (as.^2)+(bs.^2) 
    ex = cir.>9
    tau = 3./sqrt.(cir).*ex + 1-ex 
    ms = ms.*[tau;1].*[1;tau]    
    if findfirst(xs.>x) == 0
        ind_u = 2
    elseif findfirst(xs.>x) == 1
        ind_u = 2
    else
        ind_u = findfirst(xs.>x)
    end      
    x_u = xs[ind_u] 
    x_l = xs[ind_u-1] 
    h = x_u-x_l
    t = (x-x_l)./h
    fx = ys[ind_u-1]*(2*t^3-3*t^2+1) + h*ms[ind_u-1]*(t^3-2*t^2+t) + ys[ind_u]*(-2*t^3+3*t^2) + h*ms[ind_u]*(t^3-t^2)

    return fx
end

function Berkowitz(pit)
    Zb = zeros(length(pit))
    pit[pit.>=1] =0.9999999
    pit[pit.<=0] =0.0000001
    for t in collect(1:length(pit)) 
        Zb[t] = norminvcdf(abs.(pit[t]))
    end
    naux = length(Zb)-1
    mu_hat = sum(Zb)/naux

    Xaux = Zb[1:(length(Zb)-1)]-mu_hat 
    Yaux = Zb[2:length(Zb)]-mu_hat

    rho_hat = Xaux'Yaux/(Xaux'Xaux)
    eps_hat = Yaux-rho_hat*Xaux
    s2_hat = sum(eps_hat.^2)/naux

    LLu = -1/2*naux*log(s2_hat) - 1/2/s2_hat*eps_hat'eps_hat
    LLr = - 1/2*Yaux'Yaux
    LR = -2*(LLr-LLu)
    return [LR,ccdf(Chisq(3), LR)]
end


function GonzalezRivera(pit,alpha_c,alpha_main,k_main)
    # alpha_main    -- alpha-contour for k-aggregated test
    # k_main        -- lag for alpha-aggregated test
        
    n_a = length(alpha_c)   # collection of alpha-contours
    K_c=collect(1:n_a)
    n_k = length(K_c)                   # collection of lags for alpha-aggregated test
    Lambda_a = zeros(n_k,n_k)
    L_a =  zeros(n_k,1)
    report1=zeros(n_a,2,n_k)
    report2=zeros(1,2)
    report3=zeros(1,2)
    for k in 1:n_k
        alpha_contour_hat = zeros(length(alpha_c),1)
        PITk = [pit[1:(length(pit)-k)] pit[(1+k):length(pit)]]
        Tk = size(PITk)[1]
        Ind = zeros(Tk,n_a)
        Omega_k = zeros(n_a,n_a)

        for i in 1:n_a
            Indi = 1*(PITk.<sqrt(alpha_c[i]))
            Ind[:,i] = Indi[:,1].*Indi[:,2]; 
            alpha_contour_hat[i] = sum(Ind[:,i])/Tk;

            for j in 1:(i-1)
                    Omega_k[i,j] = alpha_c[j].*(1-alpha_c[i])+2*alpha_c[j]*alpha_c[i]^0.5.*(1-alpha_c[i]^0.5)
            end
            Omega_k = Omega_k + Omega_k'
            Omega_k[i,i] = alpha_c[i].*(1-alpha_c[i])+2*alpha_c[i]^1.5.*(1-alpha_c[i]^0.5)
        end

        for j in 1:n_k
            Lambda_a[j,k] = 4*alpha_main^1.5*(1-alpha_main^0.5)
        end
        Lambda_a[k,k] = alpha_main.*(1-alpha_main)+2*alpha_main^1.5.*(1-alpha_main^0.5)
        C_k = sqrt(Tk)*(alpha_contour_hat-alpha_c)
        t_c = C_k./sqrt.(alpha_c.*(1-alpha_c)+2*alpha_c.^1.5.*(1-alpha_c.^0.5))
        C_k = C_k'inv(Omega_k)*C_k

        Indi = 1*(PITk.<sqrt(alpha_main))
        Ind = Indi[:,1].*Indi[:,2]         
        alpha_main_hat = sum(Ind)/Tk

        L_a[k] = sqrt(Tk)*(alpha_main_hat-alpha_main)

        # report individual k stats and p-vals
        report1[:,:,k]=[t_c ccdf.(Chisq(1),(t_c.^2))]

        # report alpha-aggregated
        if k == k_main
            report2=[C_k ccdf.(Chisq(n_a), C_k)]
        end

    end
    # report lag-aggregated 
    L_a = L_a'inv(Lambda_a)*L_a;

    report3=[L_a ccdf.(Chisq(n_k), L_a)]
    
    return [report1,report2,report3]
end

## NUMERICAL INTEGRATION ROUTINES  
        
function ComputeInt(r,rets,cuts,ext_probs,n)
    
    a = 2*minimum(rets) 
    b = 2*maximum(rets)
    x = cos.(pi*(2*collect(1:1:n)-1)./2./n)
    S = 0

    for i in 1:n
        y=(x[i]+1)*(b-a)/2+a
        CDFy = Interpol([2*minimum(rets);cuts;2*maximum(rets)],ext_probs,y)
        S = S + ((CDFy-1*(y.>=r)).^2)*sqrt(1-x[i]^2)
    end
            
    return S*pi*(b-a)/2/n
end


