Projet Python permettant de simuler et d'analyser differentes strategies de couverture pour un Call EU dans le cadre du modele de Black-Scholes.
Il inclut des simulations de Monte Carlo pour evaluer le PnL, sa répartition et la VaR.

1- Delta Hedging Dynamique :
   Simulation en temps discret avec visualisation de l'evolution du portefeuille
   - Impact de la frequence de trading (rebalancement)
   - Modeles de volatilite stochastique

2- Gamma Hedging

3. Modele de Leland : 
   Strategie incluant les couts de transaction.

Utilisation simples :
   DeltaHedging(choix="simple", nt=0, vol=0, graph=True)
   DeltaHedging(choix="trading", nt=25, vol=0, graph=True)
   DeltaHedging(choix="vol", nt=0, vol=1, graph=True)
   DeltaHedging(choix="vol", nt=0, vol=2, graph=True)

Analyser la distribution du PnL (Monte Carlo) :
   PnL_Global(choix="vol", Nmc=1000, alpha=0.05)

Gamma Hedging:
   GammaHedging()
  
Couverture avec frais de transaction : 
  HedgingLeland()
