import matplotlib.pyplot as plt
import random as rd
import numpy as np
import math
from scipy.stats import norm

S0=1
r=0.05
sigma=0.5
T=5
Nval=100
K=1.5
B0=1
 
def d1(S,K,r,sigma,T,t):
    if t == T:
        return np.inf
    else:
        d1=(np.log(S/K)+(r+sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t));
    return d1
 
def d2(S_0,K,r,sigma,T,t):
    if t == T:
        return np.inf
    else:
        d2=(np.log(S_0/K)+(r-sigma**2/2)*(T-t))/(sigma*np.sqrt(T-t));
    return d2
 
def N(x):
    return (1/2)*(1+math.erf(x/np.sqrt(2)))
 
def Valeur_callBS(S,K,r,sigma,T,t):
    if t == T:
        return max(S-K,0)
    else:
        V = S*N(d1(S,K,r,sigma,T,t))-K*np.exp(-r*(T-t))*N(d2(S,K,r,sigma,T,t))
        return V

 
def Delta(S,K,r,sigma,T,t):
    if t == T:
        return 1.0 if S > K else 0.0
    return N(d1(S, K, r, sigma, T, t))

def DeltaHedging(choix, nt, vol, graph):
    """    
    Args:
        choix (str): 'simple', 'trading', ou 'vol'
        nt (int): Nombre de périodes de trading (utilisé si choix='trading')
        vol (int): 1 ou 2, choix du modèle de volatilité (utilisé si choix='vol')
        graph (bool) : True : renvoie les graphiques , False, renvoie les valeurs
    """
    S=np.zeros(Nval)
    P=np.zeros(Nval)
    Pactu=np.zeros(Nval)
    B=np.zeros(Nval)
    A=np.zeros(Nval)
    Vbs=np.zeros(Nval)
    dt= T/(Nval - 1)
    t=np.linspace(0,T,Nval)
    if choix == "simple":
        S[0]=S0
        Vbs[0] = Valeur_callBS(S[0],K,r,sigma,T,t[0])
        P[0] = Vbs[0] 
        A[0] = Delta(S[0],K,r,sigma,T,t[0])
        B[0] = P[0] - A[0]*S[0] 
        Pactu[0] = P[0]
        for i in range(1, Nval):
            S[i]=S[i-1]*np.exp((r-(sigma**2/2))*dt+np.sqrt(dt)*sigma*np.random.randn())
            P[i]=A[i-1]*S[i]+(1+r*dt)*B[i-1]
            A[i]=Delta(S[i],K,r,sigma,T,t[i])
            B[i]=P[i]-A[i]*S[i]
            Pactu[i]=P[i]+(Pactu[0]-P[0])*np.exp(r*t[i])
            Vbs[i]=Valeur_callBS(S[i],K,r,sigma,T,t[i])

    elif choix == "trading":
        S[0]=S0
        Vbs[0] = Valeur_callBS(S[0],K,r,sigma,T,t[0])
        P[0] = Vbs[0]
        A[0] = Delta(S[0],K,r,sigma,T,t[0])
        B[0] = P[0] - A[0]*S[0]
        Pactu[0] = P[0]

        if nt <= 0: nt = 1
        periode = Nval // nt #période de rebalancement
        
        for i in range(1,Nval):
            S[i]=S[i-1]*np.exp((r-(sigma**2/2))*dt+np.sqrt(dt)*sigma*np.random.randn())
            if (i%periode==0):
                P[i]=A[i-1]*S[i]+(1+r*dt)*B[i-1]
                A[i]=Delta(S[i],K,r,sigma,T,t[i])
                B[i]=P[i]-A[i]*S[i]
                Pactu[i]=P[i]+(Pactu[0]-P[0])*np.exp(r*t[i])
                Vbs[i]=Valeur_callBS(S[i],K,r,sigma,T,t[i])
            else:
                P[i]=A[i-1]*S[i]+(1+r*dt)*B[i-1]
                A[i]=A[i-1]
                B[i]=P[i]-A[i]*S[i]
                Pactu[i]=P[i]+(Pactu[0]-P[0])*np.exp(r*t[i])
                Vbs[i]=Valeur_callBS(S[i],K,r,sigma,T,t[i])
    elif choix=="vol":
        if vol==1:
            rand_val = np.random.rand()
            if rand_val > 0.12:
                sigma_vol = 0.5
            else:
                sigma_vol = 0.3
            
            S[0]=S0
            Vbs[0] = Valeur_callBS(S[0],K,r,sigma_vol,T,t[0])
            P[0] = Vbs[0]
            A[0] = Delta(S[0],K,r,sigma_vol,T,t[0])
            B[0] = P[0] - A[0]*S[0]
            Pactu[0] = P[0]
            sigmaL = np.zeros(Nval)
            sigmaL[0] = sigma_vol
            
            for i in range(1, Nval):
                rand_val=np.random.rand()
                if rand_val>0.12:
                    sigma_vol=0.5
                else:
                    sigma_vol=0.3
                
                S[i]=S[i-1]*np.exp((r-(sigma_vol**2/2))*dt+np.sqrt(dt)*sigma_vol*np.random.randn())
                P[i]=A[i-1]*S[i]+(1+r*dt)*B[i-1]
                A[i]=Delta(S[i],K,r,sigma_vol,T,t[i])
                B[i] = P[i]-A[i]*S[i]
                Pactu[i]=P[i]+(Pactu[0]-P[0])*np.exp(r*t[i])
                Vbs[i]=Valeur_callBS(S[i],K,r,sigma_vol,T,t[i])
                sigmaL[i]=sigma_vol
            
            if graph==True: 
                
                plt.figure(figsize=(8,5))
                plt.plot(t, sigmaL, label='Volatilité instantanée sigma(t)', linewidth=2)
                plt.xlabel("Temps t")
                plt.ylabel("Valeur de sigma")
                plt.title("Modèle 1  Évolution de la volatilité")
                plt.legend()
                plt.grid(True)
                plt.show()

        elif vol == 2:
            rand_val=np.random.rand()
            sigma1=0.3
            sigma2=0.5
            if rand_val>0.01:
                sigma_vol=0.3
            else:
                sigma_vol=0.5
            
            S[0]=S0
            Vbs[0] = Valeur_callBS(S[0],K,r,sigma_vol,T,t[0])
            P[0] = Vbs[0]
            A[0] = Delta(S[0],K,r,sigma_vol,T,t[0])
            B[0] = P[0] - A[0]*S[0]
            Pactu[0] = P[0]
            
            sigmaL=np.zeros(Nval)
            sigmaL[0]=sigma_vol
            for i in range(1,Nval):
                rand_val = np.random.rand()
                if rand_val<0.1:
                    if sigma_vol==sigma1:
                        sigma_vol=sigma2
                    else:
                        sigma_vol=sigma1
                
                S[i]=S[i-1]*np.exp((r-(sigma_vol**2/2))*dt+np.sqrt(dt)*sigma_vol*np.random.randn())
                P[i]=A[i-1]*S[i]+(1+r*dt)*B[i-1]
                A[i]=Delta(S[i],K,r,sigma_vol,T,t[i])
                B[i]=P[i]-A[i]*S[i]
                Pactu[i]=P[i]+(Pactu[0]-P[0])*np.exp(r*t[i])
                Vbs[i]=Valeur_callBS(S[i],K,r,sigma_vol,T,t[i])
                sigmaL[i]=sigma_vol
                
            if graph==True:
            
                plt.figure(figsize=(8,5))
                plt.plot(t, sigmaL, label='Volatilité instantanée sigma(t)', linewidth=2)
                plt.xlabel("Temps t")
                plt.ylabel("Valeur de sigma")
                plt.title("Modèle 2  Évolution de la volatilité")
                plt.legend()
                plt.grid(True)
                plt.show()
        else:
            print("Erreur , vol doit être égal à 1 ou 2")
            return

    else:
        print("Erreur , choix doit être 'simple', 'trading', ou 'vol'")
    
    if graph==True:
        plt.figure(figsize=(8,5))
        plt.plot(t, Vbs, label='Valeur du call (Black-Scholes)', linewidth=2)
        plt.plot(t, Pactu, label='Portefeuille actualisé', linestyle='--', linewidth=2)
        plt.xlabel("Temps t")
        plt.ylabel("Valeur")
        plt.title("Évolution du call et du portefeuille de couverture actualisé")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(8,5))
        plt.plot(t, A, label='Ratio A (Delta)', linewidth=2)
        plt.plot(t, B, label='Cash B', linestyle='--', linewidth=2)
        plt.xlabel("Temps t")
        plt.ylabel("Valeur")
        plt.title("Évolution du ratio A (delta) et du cash B")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        erreur = Pactu - Vbs
        plt.figure(figsize=(8,5))
        plt.plot(t, erreur, label="Erreur de réplication", color='red')
        plt.xlabel("Temps t")
        plt.ylabel("Erreur")
        plt.title("Erreur de réplication en fonction du temps")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    else: 
        return Vbs[-1],P[-1]

def PnL_Global(choix, Nmc, alpha):
    
    if choix == "simple":
        valPnL = []
        for i in range(Nmc):
            VbsFin, PactuFin = DeltaHedging('simple', 0, 0, False)
            valPnL.append(PactuFin - VbsFin)
        
        moy = np.mean(valPnL)
        print('L espérance du PnL est de : ', moy)

        valPnL = np.sort(valPnL)
        cdf = np.arange(1, len(valPnL)+1) / len(valPnL)

        a = 0.3
        nbins = 100
        width = (2*a) / nbins
        t = np.linspace(-a + width/2, a - width/2, nbins)
        hist = np.zeros(nbins)

        for j in range(len(valPnL)):
            idx = int(np.floor((valPnL[j] + a) / width))
            if 0 <= idx < nbins:
                hist[idx] += 1
            elif valPnL[j] == a:
                hist[-1] += 1
        if hist.sum() > 0:
            hist = hist / hist.sum()
        
        # VaR    
        k = int(alpha * len(valPnL))
        if k >= len(valPnL): k = len(valPnL) - 1
        print('La VaR est de :', valPnL[k])

        plt.figure(figsize=(12, 5))
        

        Esp = np.mean(valPnL)
        Var = np.var(valPnL)
        ksi = np.zeros(Nmc)
        Z = np.zeros(Nmc)
        V = np.zeros(Nmc)
        N01 = np.sort(np.random.randn(Nmc))

        for n in range(Nmc):
            ksi[n] = (valPnL[n] - Esp) / np.sqrt(Var)
            V[n] = (n + 0.5) / Nmc
            Z[n] = norm.ppf(V[n])
            
        plt.plot(N01, np.sort(ksi), label="Données")
        plt.plot(N01, N01, label="Référence Normale", linestyle='--')
        plt.title("QQ-Plot")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(valPnL, cdf, lw=2)
        plt.title("Fonction de répartition empirique du PnL")
        plt.xlabel("PnL")
        plt.ylabel("F(PnL)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.bar(t, hist, width=width, edgecolor='black', alpha=0.7)
        plt.title("Histogramme empirique du PnL")
        plt.xlabel("PnL")
        plt.ylabel("Fréquence relative")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif choix == "trading":
        valPnL1 = []
        valPnL2 = []
        valPnL3 = []

        for i in range(Nmc):
            VbsFin1, PactuFin1 = DeltaHedging('trading', 100, 0, False)
            valPnL1.append(PactuFin1 - VbsFin1)
            
            VbsFin2, PactuFin2 = DeltaHedging('trading', 25, 0, False)
            valPnL2.append(PactuFin2 - VbsFin2)
            
            VbsFin3, PactuFin3 = DeltaHedging('trading', 5, 0, False)
            valPnL3.append(PactuFin3 - VbsFin3)
            
        moy1 = np.mean(valPnL1)
        moy2 = np.mean(valPnL2)
        moy3 = np.mean(valPnL3)
        
        var1 = np.var(valPnL1, ddof=1)
        var2 = np.var(valPnL2, ddof=1)
        var3 = np.var(valPnL3, ddof=1)
        
        print(f'L espérance du PnL est de : {moy1:.4f} (Nt=100), {moy2:.4f} (Nt=25), {moy3:.4f} (Nt=5)')
        print(f'La variance du PnL est de : {var1:.4f} (Nt=100), {var2:.4f} (Nt=25), {var3:.4f} (Nt=5)')
        
        valPnL1 = np.sort(valPnL1)
        valPnL2 = np.sort(valPnL2)
        valPnL3 = np.sort(valPnL3)
        
        cdf1 = np.arange(1, len(valPnL1)+1) / len(valPnL1)
        cdf2 = np.arange(1, len(valPnL2)+1) / len(valPnL2)
        cdf3 = np.arange(1, len(valPnL3)+1) / len(valPnL3)

        a = 0.3
        nbins = 100
        width = (2*a) / nbins
        t = np.linspace(-a + width/2, a - width/2, nbins)
        
        def compute_hist(data, a, width, nbins):
            h = np.zeros(nbins)
            for val in data:
                idx = int(np.floor((val + a) / width))
                if 0 <= idx < nbins:
                    h[idx] += 1
                elif val == a:
                    h[-1] += 1
            if h.sum() > 0: h /= h.sum()
            return h
        
        hist1 = compute_hist(valPnL1, a, width, nbins)
        hist2 = compute_hist(valPnL2, a, width, nbins)
        hist3 = compute_hist(valPnL3, a, width, nbins)
        

        k = int(alpha * len(valPnL2))
        VaR = valPnL2[k]
        print('La VaR (pour Nt=25) est de :', VaR)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(valPnL1, cdf1, lw=2, label="Nt=100")
        plt.axvline(x=VaR, color='r', linestyle='--', lw=2, label=f'VaR Nt=25')
        plt.axhline(y=alpha, color='gray', linestyle=':', lw=1)
        plt.plot(valPnL2, cdf2, lw=2, label="Nt=25")
        plt.plot(valPnL3, cdf3, lw=2, label="Nt=5")
        plt.title("Fonction de répartition empirique du PnL")
        plt.xlabel("PnL")
        plt.ylabel("F(PnL)")
        plt.legend()
        plt.grid(True)
        plt.xlim(-1.5, 1.5) 

        plt.subplot(1, 2, 2)
        plt.bar(t, hist1, width=width, edgecolor='blue', alpha=0.3, label="Nt=100")
        plt.bar(t, hist2, width=width, edgecolor='orange', alpha=0.3, label="Nt=25")
        plt.bar(t, hist3, width=width, edgecolor='green', alpha=0.3, label="Nt=5")
        plt.title("Histogramme empirique du PnL")
        plt.xlabel("PnL")
        plt.ylabel("Fréquence relative")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    elif choix == "vol":
        valPnL1 = []
        valPnL2 = []
        
        for i in range(Nmc):
            VbsFin1, PactuFin1 = DeltaHedging("vol", 0, 1, False)
            valPnL1.append(PactuFin1 - VbsFin1)
            
            VbsFin2, PactuFin2 = DeltaHedging("vol", 0, 2, False)
            valPnL2.append(PactuFin2 - VbsFin2)
            
        moy1 = np.mean(valPnL1)
        moy2 = np.mean(valPnL2)
        
        print(f'L espérance du PnL est de : {moy1:.4f} (Modèle 1) et {moy2:.4f} (Modèle 2)')
        
        valPnL1 = np.sort(valPnL1)
        valPnL2 = np.sort(valPnL2)
        
        cdf1 = np.arange(1, len(valPnL1)+1) / len(valPnL1)
        cdf2 = np.arange(1, len(valPnL2)+1) / len(valPnL2)
        
        a = 0.3
        nbins = 100
        width = (2*a) / nbins
        t = np.linspace(-a + width/2, a - width/2, nbins)
        
        def compute_hist(data, a, width, nbins):
            h = np.zeros(nbins)
            for val in data:
                idx = int(np.floor((val + a) / width))
                if 0 <= idx < nbins:
                    h[idx] += 1
                elif val == a:
                    h[-1] += 1
            if h.sum() > 0: h /= h.sum()
            return h
        
        hist1 = compute_hist(valPnL1, a, width, nbins)
        hist2 = compute_hist(valPnL2, a, width, nbins)
        k = int(alpha * len(valPnL1))
        VaR = valPnL1[k]
        print('La VaR (Modèle 1) est de :', VaR)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(valPnL1, cdf1, lw=2, label="Modèle 1")
        plt.plot(valPnL2, cdf2, lw=2, label="Modèle 2")
        plt.title("Fonction de répartition empirique du PnL")
        plt.xlabel("PnL")
        plt.ylabel("F(PnL)")
        plt.legend()
        plt.grid(True)
        plt.xlim(-0.5, 0.5)
        
        plt.subplot(1, 2, 2)
        plt.bar(t, hist1, width=width, edgecolor='blue', alpha=0.4, label="Modèle 1")
        plt.bar(t, hist2, width=width, edgecolor='orange', alpha=0.4, label="Modèle 2")
        plt.title("Histogramme empirique du PnL")
        plt.xlabel("PnL")
        plt.ylabel("Fréquence relative")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    else:
        print("Erreur : choix = 'simple', 'trading', ou 'vol'")
  
def Gamma(S,K,r,sigma,T,t):
    Gamma=(np.exp(-(1/2)*d1(S,K,r,sigma,T,t)**2)/(S*sigma*np.sqrt(2*np.pi*(T-t))))
    return Gamma

def GammaHedging():
    S0=1 
    r=0.05
    sigma=0.5
    Nval=100
    T1=5 #maturité call à couvrir
    T2=10 #maturité iNvalstrument de couverture
    dt=T1/Nval
    K=1.5 #K1=K2=K
    S=np.zeros(Nval) #val sous-jaceNvalt
    B=np.zeros(Nval)  # liste  cash 
    A=np.zeros(Nval) # liste actions
    t=np.linspace(0,T1,Nval) #discrétisation maturité 1
    P=np.zeros(Nval) #liste portefeuille
    G=np.zeros(Nval)
    VbsC=np.zeros(Nval) #valeur B&S call de couverture
    VbsV=np.zeros(Nval) #valeur B&S call à couvrir
    Pactu=np.zeros(Nval)
    S[0]=S0
    G0=Gamma(S[0],K,r,sigma,T1,0)/Gamma(S[0],K,r,sigma,T2,0)
    G[0]=G0
    A[0]=N(d1(S[0],K,r,sigma,T1,0))-G[0]*N(d1(S[0],K,r,sigma,T2,0))
    B[0]=1 
    VbsC[0]=Valeur_callBS(S[0],K,r,sigma,T2,t[0])
    VbsV[0]=Valeur_callBS(S[0],K,r,sigma,T1,t[0])
    P[0]=A[0]*S[0]+G[0]*VbsC[0]+B[0]
    Pactu[0]=VbsV[0]
    for i in range(1,Nval):
        S[i]=S[i-1]*np.exp((r-(sigma**2)/2)*dt+np.sqrt(dt)*sigma*np.random.randn())
        VbsC[i]=Valeur_callBS(S[i],K,r,sigma,T2,t[i])
        VbsV[i]=Valeur_callBS(S[i],K,r,sigma,T1,t[i])
        P[i]=A[i-1]*S[i]+G[i-1]*VbsC[i]+B[i-1]*(1+r*dt)
        G[i]=Gamma(S[i],K,r,sigma,T1,t[i])/Gamma(S[i],K,r,sigma,T2,t[i])
        A[i]=N(d1(S[i],K,r,sigma,T1,t[i]))-G[i]*N(d1(S[i],K,r,sigma,T2,t[i]))
        B[i]=P[i]-A[i]*S[i]-G[i]*VbsC[i]
        Pactu[i]=P[i]+(VbsV[0]-P[0])*np.exp(r*t[i])
    
    plt.figure(figsize=(8,5))
    plt.plot(t, VbsV, label='Valeur du call (Black-Scholes)', linewidth=2)
    plt.plot(t, Pactu, label='Portefeuille actualisé', linestyle='--', linewidth=2)
    plt.xlabel("Temps t")
    plt.ylabel("Valeur")
    plt.title("Évolution du call et du portefeuille de couverture actualisé")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8,5))
    plt.plot(t, A, label='Ratio A (Delta)', linewidth=2)
    plt.plot(t, G, label='Ratio G (Gamma)', linewidth=2)
    plt.plot(t, B, label='Cash B', linestyle='--', linewidth=2)
    plt.xlabel("Temps t")
    plt.ylabel("Valeur")
    plt.title("Évolution du ratio A (delta), du ratio G (Gamma) et du cash B")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    erreur = Pactu - VbsV
    plt.figure(figsize=(8,5))
    plt.plot(t, erreur, label="Erreur de réplication", color='red')
    plt.xlabel("Temps t")
    plt.ylabel("Erreur")
    plt.title("Erreur de réplication en fonction du temps")
    plt.legend()
    plt.grid(True)
    plt.show()
        
def HedgingLeland():
    Nval=1040
    T=1
    S=np.zeros(Nval) #déclaration liste valeurs sous-jacent
    P=np.zeros(Nval) #déclaration liste valeurs portefeuille
    B=np.zeros(Nval)  #déclaration liste valeurs cash 
    A=np.zeros(Nval) #déclaration liste nombre de sous-jacent détenus 
    Vbs=np.zeros(Nval)  #déclaration liste valeurs théoriques du call 
    t=np.linspace(0,1,Nval) # discrétisation de la période
    dt=1/(Nval-1)
    S0=100
    sigma=0.25
    r=0.05
    k=0.01
    K=100
    sigma=sigma*np.sqrt(1+(k/sigma)*np.sqrt(2/(np.pi*dt)))
    S[0]=S0
    A[0]=Delta(S[0], K, r, sigma, T,0)
    Vbs[0]=Valeur_callBS(S[0], K, r, sigma, T, 0)
    B[0]=Vbs[0]-A[0]*S[0]-k*np.abs(A[0])*S[0]
    P[0]=A[0]*S[0]+B[0]
    for i in range(1,len(t)):
        S[i]=S[i-1]*np.exp((r-(sigma**2/2))*dt+np.sqrt(dt)*sigma*np.random.randn())
        A[i]=Delta(S[i], K, r, sigma, T,t[i])
        B[i]=B[i-1]*(1+r*dt)-k*np.abs(A[i]-A[i-1])*S[i]-(A[i]-A[i-1])*S[i]
        P[i]=A[i-1]*S[i]+B[i]
        Vbs[i]=Valeur_callBS(S[i], K, r, sigma, T, t[i])
        
    plt.figure(figsize=(8,5))
    plt.plot(t, Vbs, label='Valeur du call (Black-Scholes)', linewidth=2)
    plt.plot(t, P, label='Portefeuille actualisé', linestyle='--', linewidth=2)
    plt.xlabel("Temps t")
    plt.ylabel("Valeur")
    plt.title("Évolution du call et du portefeuille de couverture actualisé")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    erreur = P - Vbs
    plt.figure(figsize=(8,5))
    plt.plot(t, erreur, label="Erreur de réplication", color='red')
    plt.xlabel("Temps t")
    plt.ylabel("Erreur")
    plt.title("Erreur de réplication en fonction du temps")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def LelandResultats():
    Nval=1040
    T=1
    S=np.zeros(Nval) #déclaration liste valeurs sous-jacent
    P=np.zeros(Nval) #déclaration liste valeurs portefeuille
    B=np.zeros(Nval)  #déclaration liste valeurs cash 
    A=np.zeros(Nval) #déclaration liste nombre de sous-jacent détenus 
    Vbs=np.zeros(Nval)  #déclaration liste valeurs théoriques du call 
    t=np.linspace(0,1,Nval) # discrétisation de la période
    dt=1/(Nval-1)
    S0=100
    sigma=0.25
    r=0.05
    k=0.01
    K=100
    sigma=sigma*np.sqrt(1+(k/sigma)*np.sqrt(2/(np.pi*dt)))
    S[0]=S0
    A[0]=Delta(S[0], K, r, sigma, T,0)
    Vbs[0]=Valeur_callBS(S[0], K, r, sigma, T, 0)
    B[0]=Vbs[0]-A[0]*S[0]-k*np.abs(A[0])*S[0]
    P[0]=A[0]*S[0]+B[0]
    for i in range(1,len(t)):
        S[i]=S[i-1]*np.exp((r-(sigma**2/2))*dt+np.sqrt(dt)*sigma*np.random.randn())
        A[i]=Delta(S[i], K, r, sigma, T,t[i])
        B[i]=B[i-1]*(1+r*dt)-k*np.abs(A[i]-A[i-1])*S[i]-(A[i]-A[i-1])*S[i]
        P[i]=A[i-1]*S[i]+B[i]
        Vbs[i]=Valeur_callBS(S[i], K, r, sigma, T, t[i])
    return Vbs[-1],P[-1]
        

def PnLLeland(Nmc):
    PactuFin = 0
    VbsFin = 0
    moy = 0
    valPnL = []
    for i in range(Nmc):
        VbsFin, PactuFin = LelandResultats()
        valPnL.append(PactuFin - VbsFin)
        moy += valPnL[-1]
    moy=moy/Nmc
    print('L espérance du PnL est de : ', moy)

    valPnL=np.sort(valPnL)
    cdf=np.arange(1, len(valPnL)+1) / len(valPnL)

    a=40
    nbins=100

    width = (2*a) / nbins

    t = np.linspace(-a + width/2, a - width/2, nbins)
    hist = np.zeros(nbins)

    for j in range(len(valPnL)):
        #indice de bin 
        idx = int(np.floor((valPnL[j] + a) / width))
        if 0 <= idx < nbins:
            hist[idx] += 1
        elif valPnL[j] == a:
            hist[-1] += 1
    if hist.sum() > 0:
        hist = hist / hist.sum()
    
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(valPnL, cdf, lw=2)
    plt.title("Fonction de répartition empirique du PnL")
    plt.xlabel("PnL")
    plt.ylabel("F(PnL)")
    plt.grid(True)

    # Histogramme
    plt.subplot(1, 2, 2)
    plt.bar(t, hist, width=width, edgecolor='black', alpha=0.7)
    plt.title("Histogramme empirique du PnL")
    plt.xlabel("PnL")
    plt.ylabel("Fréquence relative")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    
def HedgingLeland2():
    Nval=1040
    T=1
    S=np.zeros(Nval) #déclaration liste valeurs sous-jacent
    P=np.zeros(Nval) #déclaration liste valeurs portefeuille
    B=np.zeros(Nval)  #déclaration liste valeurs cash 
    A=np.zeros(Nval) #déclaration liste nombre de sous-jacent détenus 
    Vbs=np.zeros(Nval)  #déclaration liste valeurs théoriques du call 
    t=np.linspace(0,1,Nval) # discrétisation de la période
    dt=1/(Nval-1)
    S0=100
    sigma=0.25
    r=0.05
    k=0.01
    K=100
    sigma=sigma*np.sqrt(1+(k/sigma)*np.sqrt(2/(np.pi*dt)))
    S[0]=S0
    A[0]=Delta(S[0], K, r, sigma, T,0)
    Vbs[0]=Valeur_callBS(S[0], K, r, sigma, T, 0)
    B[0]=Vbs[0]-A[0]*S[0]-k*np.abs(A[0])*S[0]
    P[0]=A[0]*S[0]+B[0]
    u=0.01
    for i in range(1,len(t)):
        S[i]=S[i-1]*np.exp((r-(sigma**2/2))*dt+np.sqrt(dt)*sigma*np.random.randn())
    for i in range(1,len(t)):
        if (np.log(S[i]/S[i-1])>u) or (np.log(S[i]/S[i-1])<-u):
            A[i]=Delta(S[i], K, r, sigma, T,t[i])
            B[i]=B[i-1]*(1+r*dt)-k*np.abs(A[i]-A[i-1])*S[i]-(A[i]-A[i-1])*S[i]
            P[i]=A[i-1]*S[i]+B[i]
            Vbs[i]=Valeur_callBS(S[i], K, r, sigma, T, t[i])
        else:
            A[i]=A[i-1]
            B[i]=B[i-1]*(1+r*dt)
            P[i]=A[i-1]*S[i]+B[i]
            Vbs[i]=Valeur_callBS(S[i], K, r, sigma, T, t[i])
        
    plt.figure(figsize=(8,5))
    plt.plot(t, Vbs, label='Valeur du call (Black-Scholes)', linewidth=2)
    plt.plot(t, P, label='Portefeuille actualisé', linestyle='--', linewidth=2)
    plt.xlabel("Temps t")
    plt.ylabel("Valeur")
    plt.title("Évolution du call et du portefeuille de couverture actualisé")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    erreur = P - Vbs
    plt.figure(figsize=(8,5))
    plt.plot(t, erreur, label="Erreur de réplication", color='red')
    plt.xlabel("Temps t")
    plt.ylabel("Erreur")
    plt.title("Erreur de réplication en fonction du temps")
    plt.legend()
    plt.grid(True)
    plt.show()

def Leland2Resultats():
    Nval=1040
    T=1
    S=np.zeros(Nval) #déclaration liste valeurs sous-jacent
    P=np.zeros(Nval) #déclaration liste valeurs portefeuille
    B=np.zeros(Nval)  #déclaration liste valeurs cash 
    A=np.zeros(Nval) #déclaration liste nombre de sous-jacent détenus 
    Vbs=np.zeros(Nval)  #déclaration liste valeurs théoriques du call 
    t=np.linspace(0,1,Nval) # discrétisation de la période
    dt=1/(Nval-1)
    S0=100
    sigma=0.25
    r=0.05
    k=0.01
    K=100
    sigma=sigma*np.sqrt(1+(k/sigma)*np.sqrt(2/(np.pi*dt)))
    S[0]=S0
    A[0]=Delta(S[0], K, r, sigma, T,0)
    Vbs[0]=Valeur_callBS(S[0], K, r, sigma, T, 0)
    B[0]=Vbs[0]-A[0]*S[0]-k*np.abs(A[0])*S[0]
    P[0]=A[0]*S[0]+B[0]
    u=0.01
    for i in range(1,len(t)):
        S[i]=S[i-1]*np.exp((r-(sigma**2/2))*dt+np.sqrt(dt)*sigma*np.random.randn())
    for i in range(1,len(t)):
        if (np.log(S[i]/S[i-1])>u) or (np.log(S[i]/S[i-1])<-u):
            A[i]=Delta(S[i], K, r, sigma, T,t[i])
            B[i]=B[i-1]*(1+r*dt)-k*np.abs(A[i]-A[i-1])*S[i]-(A[i]-A[i-1])*S[i]
            P[i]=A[i-1]*S[i]+B[i]
            Vbs[i]=Valeur_callBS(S[i], K, r, sigma, T, t[i])
        else:
            A[i]=A[i-1]
            B[i]=B[i-1]*(1+r*dt)
            P[i]=A[i-1]*S[i]+B[i]
            Vbs[i]=Valeur_callBS(S[i], K, r, sigma, T, t[i])
    return Vbs[-1],P[-1]

def PnLLeland2(Nmc):
    PactuFin = 0
    VbsFin = 0
    moy = 0
    valPnL = []
    for i in range(Nmc):
        VbsFin, PactuFin = Leland2Resultats()
        valPnL.append(PactuFin - VbsFin)
        moy += valPnL[-1]
    moy=moy/Nmc
    print('L espérance du PnL est de : ', moy)

    valPnL=np.sort(valPnL)
    cdf=np.arange(1, len(valPnL)+1) / len(valPnL)

    a=20
    nbins=100

    width = (2*a) / nbins

    t = np.linspace(-a + width/2, a - width/2, nbins)
    hist = np.zeros(nbins)

    for j in range(len(valPnL)):
        #indice de bin 
        idx = int(np.floor((valPnL[j] + a) / width))
        if 0 <= idx < nbins:
            hist[idx] += 1
        elif valPnL[j] == a:
            hist[-1] += 1
    if hist.sum() > 0:
        hist = hist / hist.sum()
    
    
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(valPnL, cdf, lw=2)
    plt.title("Fonction de répartition empirique du PnL")
    plt.xlabel("PnL")
    plt.ylabel("F(PnL)")
    plt.grid(True)

    # Histogramme
    plt.subplot(1, 2, 2)
    plt.bar(t, hist, width=width, edgecolor='black', alpha=0.7)
    plt.title("Histogramme empirique du PnL")
    plt.xlabel("PnL")
    plt.ylabel("Fréquence relative")
    plt.grid(True)

    plt.tight_layout()
    plt.show()