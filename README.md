# MCMC Algorithms Comparison

In this repository, we compare RWMH, HMC, NUTS, raHMC, and uHMC, as well as the dual-averaging versions of HMC, NUTS, and raHMC, through five examples:

1. Gamma(5,1) distribution
2. Multivariate normal distribution
3. Bivariate Gaussian mixture
4. Standard Cauchy distribution
5. Eight schools model

We present our implementations in three separate files: one file for the sampler functions, another file for the model functions, and a final `run_all.py` file for running the experiments.

In the `results` folder, the reader can explore the output of our experiments. The `requirements.txt` file contains the package versions used in the project.

The project was initially run in a Google Colab session and was later organized into this repository.

# Authors

Mario Alejandro Molina Palma

Saúl Abraham Granados Carmona
