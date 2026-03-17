# Optimal Execution with Stochastic Delay - Résumé Critique

Ce projet contient un résumé critique et une analyse de l'article **"Optimal Execution with Stochastic Delay"** d'Álvaro Cartea et Leandro Sánchez-Betancourt (*Finance and Stochastics*, 2023). 

Ce travail a été réalisé par **Dan Allouche**, **Liam Abensour**, et **Corentin Srun** dans le cadre du cours de Philippe Bergault (M2 MASEF - Université Paris Dauphine) pour le Cycle de Conférences : *Strategies and Actors of Portfolio Management*.

## Contenu du projet

- **`Optimal_Execution_Summary.pdf`** : Le résumé critique du papier. Il détaille le modèle avec latence stochastique, la modélisation par processus de Markov et le système HJBQVI, tout en apportant des critiques mathématiques et pratiques.
- **`Optimal_Execution_Article.pdf`** : L'article de recherche original.
- **`optimal_execution_simulation.py`** : Code Python implémentant une simulation Monte-Carlo simplifiée illustrant les mécanismes de la stratégie avec latence aléatoire (RLOS) face aux stratégies TWAP et ENOW.

## Comment relire le projet

1. Le document principal est le **`Optimal_Execution_Summary.pdf`** qui contient toute l'analyse.
2. Pour exécuter la simulation, vous pouvez lancer le script Python :
   ```bash
   python optimal_execution_simulation.py
   ```
   *Note : Le script nécessite `numpy`, `matplotlib`.*
