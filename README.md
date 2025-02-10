# notebook-gallery

QuantEcon Notebook Gallery - sharing notebooks on computational economics

## notes.quantecon.org Notebook Gallery

1. [OLS and ML estimations with Julia JuMP](ipynb/mizuhiro_suzuki-jump_estimation_basic.ipynb)
  - *Author:* Mizuhiro Suzuki
  - *Programming Language:* Julia
  - *Published:* 2025 Jan 02
  - *Summary:* I used Julia JuMP for OLS and ML estimations to show how to use JuMP with parameter estimations.
2. [Kidney exchanges with a recursive algorithm](ipynb/mizuhiro_suzuki-kidney_exchange.ipynb)
  - *Author:* Mizuhiro Suzuki
  - *Programming Language:* Julia
  - *Published:* 2024 Jan 09
  - *Summary:* Julia implementation of a recursive algorithm for kidney exchanges
3. [Deep Learning Solutions of DSGE Models: a Technical Report](ipynb/julien_pascal-dl_dsge_quantecon.ipynb)
  - *Author:* Julien Pascal
  - *Programming Language:* Python
  - *Published:* 2023 Nov 03
  - *Summary:* This notebook illustrates how to use Deep Learning (DL) methods to solve DSGE models.  We also investigate how hyperparameter choices, such as the learning rate or the number of nodes in the hidden layer(s), affect the accuracy of the numerical solution obtained through DL.  
4. [FX Swaption Valuation using QuantLib and Monte Carlo Simulation](ipynb/amit_kumar_jha-fx_swaption_quantlib_montecarlo.ipynb)
  - *Author:* Amit Kumar Jha
  - *Programming Language:* Python
  - *Published:* 2023 Oct 16
  - *Summary:* This project focuses on the valuation of an FX swaption using QuantLib and Monte Carlo simulation. An FX swaption is a financial derivative that provides the holder the right but not the obligation to enter into a foreign exchange swap contract at a future date, at a predetermined strike rate
5. [Linear programming formulation for two-sided matching](ipynb/mizuhiro_suzuki-two_sided_lp.ipynb)
  - *Author:* Mizuhiro Suzuki
  - *Programming Language:* Julia
  - *Published:* 2023 Jul 20
  - *Summary:* I implemented the two-sided perfect stable matching with the linear programming formulation.
6. [Statistics](ipynb/zixu_wang（王梓旭）-v01_statistics_2.ipynb)
  - *Author:* Zixu Wang（王梓旭）
  - *Programming Language:* Python
  - *Published:* 2023 May 19
  - *Summary:* nan
7. [Solving Linear Complementarity Problems with QuantEcon.py](ipynb/daisuke_oyama-qe_lcp.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Python
  - *Published:* 2023 May 17
  - *Summary:* This notebook demonstrates the usage of the linear complementarity problem (LCP) solver `lcp_lemke` in `QuantEcon.py`.
8. [The stochastic optimal growth model with the collocation method + Gaussian quadrature approximation](ipynb/mizuhiro_suzuki-solving_stochastic_optimal_growth_model_with_collocation_and_quadrature_methods.ipynb)
  - *Author:* Mizuhiro Suzuki
  - *Programming Language:* Julia
  - *Published:* 2023 May 08
  - *Summary:* I implemented the collocation method and Gaussian quadrature method to solve the stochastic optimal growth model. 
9. [Calibration of Interest Rate Models: A Study on Hull-White Model with Swaption Data](ipynb/amit_kumar_jha-calibration_of_interest_rate_models_a_study_on_hull_white_model_with_swaption_data.ipynb)
  - *Author:* Amit Kumar Jha
  - *Programming Language:* Python
  - *Published:* 2023 Apr 19
  - *Summary:* Implementation of the Hull-White one-factor model to calibrate a term structure of interest rates to a set of swaption volatilities. The model is used to calculate the price of European swaptions and compare them to market prices, in order to find the best values of the model parameters that minimize the differences between model and market prices.
10. [Evaluating the effectiveness of macroeconomic policies through computer simulations](ipynb/amit_kumar_jha-evaluating_the_effectiveness_of_macroeconomic_policies_through_computer_simulations.ipynb)
  - *Author:* Amit Kumar Jha
  - *Programming Language:* Python
  - *Published:* 2023 Feb 04
  - *Summary:* Macroeconomic policies are the measures taken by governments to manage the economy, and computer simulations provide a tool for analyzing their potential impact. By simulating the behavior of the economy under different scenarios, economists can gain a deeper understanding of how different policies may impact the economy. These simulations can also be used to evaluate the impact of a policy on different segments of the population and compare the outcomes of different policy scenarios. However, computer simulations have limitations, including that they are based on assumptions that may not always be accurate and do not take into account all the factors that can influence the economy. As a result, the results of simulations should be interpreted with caution and used in conjunction with other sources of information in policymaking.
11. [India 10Y Yield Forecast Using Kalman Filters](ipynb/amit_kumar_jha-india_10y_yield_forecast_using_kalman_filters.ipynb)
  - *Author:* Amit Kumar Jha
  - *Programming Language:* Python
  - *Published:* 2022 Dec 25
  - *Summary:* The Kalman filter is a mathematical tool used to estimate the state of a system based on noisy measurements. It is based on Bayesian filtering, which involves using Bayes' theorem to update the probability of a hypothesis as new evidence becomes available.  At each time step, the Kalman filter makes a prediction about the state of the system based on the previous estimate and the dynamics of the system. It then compares this prediction to the actual measurement of the system, taking into account the measurement noise. 
12. [Using LP to solve a portfolio optimization problem with stochastic dominance constraints](ipynb/nikolai_chow-qe_notebook_-_optimization_with_stochastic_dominance_constraints.ipynb)
  - *Author:* Nikolai Chow
  - *Programming Language:* Python
  - *Published:* 2022 Dec 19
  - *Summary:* In this notebook, we will examine the concept of stochastic dominance and its application in a portfolio optimization problem. Specifically, we will look at a method proposed by Dentcheva and Ruszczyński (2003) that incorporates stochastic dominance constraints into the optimization process. Using Python, we will demonstrate the use of this method with an example from Dentcheva and Ruszczyński (2003). 
13. [Explaining_DSGE_Model_with_Python_Implementation](ipynb/amit_kumar_jha-explaining_dsge_model_with_python_implementation_.ipynb)
  - *Author:* Amit Kumar Jha
  - *Programming Language:* Python
  - *Published:* 2022 Dec 18
  - *Summary:* In this notebook, you'll learn about DSGE Model in macroeconomics and how to implement it in Python
14. [Pricing a European call option: CPU vs GPU and Numba vs JAX](ipynb/john_stachurski-euro_option.ipynb)
  - *Author:* John Stachurski
  - *Programming Language:* Python
  - *Published:* 2022 Nov 14
  - *Summary:* An illustration of the power of Google JAX for fast computations on the GPU, using a simple option pricing example.  The point of comparison is a basic Numba-CUDA implementation.  This Numba version is inefficient and can be accelerated with careful thought.  But JAX takes care of optimal parallelization for us, on the fly.  TL/DR: Unless you have lots of spare time, leave GPU code optimization to Google's engineers.
15. [A Primer on VARs](ipynb/mcherculano-primer_on_vars.ipynb)
  - *Author:* mcherculano
  - *Programming Language:* Matlab
  - *Published:* 2022 Nov 02
  - *Summary:* A gentle Intro to macroeconometrics with VARs.
16. [Historical_analysis_of_the_Forex_market_%28USD_INR%29_using_Python](ipynb/nan-historical_analysis_of_the_forex_market_(usd_inr)_using_python.ipynb)
  - *Author:* nan
  - *Programming Language:* Python
  - *Published:* 2022 Oct 12
  - *Summary:* The foreign exchange market has seen significant movement in recent years. The dollar has appreciated in value relative to other currencies, especially compared to the INR. In this repository we will use historical data to analyze how the dollar has changed and whether there is a relationship between the dollar and INR and Indian interest rates. The analysis will be conducted using Python. Throughout this article we will learn 1. How to get and analyze financial data 2. How Dollar-INRexchange rate have been changing 3. Relationship between the U.S.-India interest rate differential and the exchange rate
17. [Discovering Faster Matrix Multiplication Algorithms with Human Intelligence](ipynb/shu_hu-discovering_faster_matrix_multiplication_algorithms_with_human_intelligence.ipynb)
  - *Author:* Shu Hu
  - *Programming Language:* Python
  - *Published:* 2022 Oct 11
  - *Summary:* This notebook implements the naive algorithm and the Strassen algorithm for computing matrix multiplication, along with the correspondences in ``numpy`` and ``jax``.
18. [Generating Input Output Networks Using US Data](ipynb/shu_hu-generating_input_output_networks_using_us_data.ipynb)
  - *Author:* Shu Hu
  - *Programming Language:* Python
  - *Published:* 2022 Jul 22
  - *Summary:* This notebook teaches how to visualize an input-output network using US data. 
19. [Solving the Neal Career choice problem using Reinforcement Learning](ipynb/spencer_lyon-career_rl.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Julia
  - *Published:* 2022 Apr 22
  - *Summary:* # Reinforcement Learning -- Economic Example  **Prerequisites**  - Dynamic Programming - Reinforcement Learning Introduction - Reinforcement Learning Sarsa algorithm - Reinforcement Learning Q-learning algorithm  **Outcomes**  - Understand the main points of the career choice model of Derek Neal - See how career choice model can be cast as a RL problem - Use Sarsa and QLearning to "solve" career choice model  **References**  - Barto & Sutton book (online by authors [here](http://incompleteideas.net/book/the-book.html)) chapters 4-6 - [QuantEcon lecture](https://python.quantecon.org/career.html) on career choice model 
20. [Coase's Theory of the Firm -- in Julia!](ipynb/john_stachurski-coase.ipynb)
  - *Author:* John Stachurski
  - *Programming Language:* Julia
  - *Published:* 2022 Apr 07
  - *Summary:* This notebook studies a model that represents Coase's famous theory of the firm and implements numerical experiments in Julia.
21. [Replication of Rust %281987%29 using Google's JAX](ipynb/natasha-rust-jax.ipynb)
  - *Author:* Natasha
  - *Programming Language:* Python
  - *Published:* 2022 Mar 22
  - *Summary:* How much faster can I solve Rust (1987) using JAX instead of Numba? Click here to see.
22. [Replication of Rust %281987%29](ipynb/natasha-rust.ipynb)
  - *Author:* Natasha
  - *Programming Language:* Python
  - *Published:* 2022 Mar 18
  - *Summary:* This notebook replicates the main results in Rust (1987) using the raw data files. I hope that this notebook provides an easy-to-follow guide to solving the model!
23. [Dynamic Programming on the GPU via JAX](ipynb/john_stachurski-jax_dp.ipynb)
  - *Author:* John Stachurski
  - *Programming Language:* Python
  - *Published:* 2022 Mar 14
  - *Summary:* Parallelization on GPUs is one of the major trends of modern scientific computing.  In this notebook we examine a simple implementation of dynamic programming on the GPU using Python and the Google JAX library.
24. [ECON 323 Final Project: The Effect of International Players on Domestic Player Wages in the British Premier League](ipynb/maxime_rotsaert-soccer_project_(2).ipynb)
  - *Author:* Maxime Rotsaert
  - *Programming Language:* Python
  - *Published:* 2021 Dec 18
  - *Summary:* My project’s central focus is the effect of international football players on the average wage of domestic players in the British Premier League. I analyze how domestic and foreign wages are distributed across teams and whether having a greater percentage of a team that is foreign affects domestic player wages. This involves data from five datasets found on a British Premier League database (the last five seasons of team rosters and salaries) as well as data I’ve compiled on my own on team performance and season standings year on year from Sky Sports. I do several visualizations and some regressions as well. 
25. [ECON 323 Final Project - Examining the Effect of International Players on Domestic Player Wages in the Premier League](ipynb/maxime_rotsaert-soccer_project_(1).ipynb)
  - *Author:* Maxime Rotsaert
  - *Programming Language:* Python
  - *Published:* 2021 Dec 18
  - *Summary:* My project’s central focus is the effect of international football players on the average wage of domestic players in the British Premier League. I analyze how domestic and foreign wages are distributed across teams and whether having a greater percentage of a team that is foreign affects domestic player wages. This involves data from five datasets found on a British Premier League database (the last five seasons of team rosters and salaries) as well as data I’ve compiled on my own on team performance and season standings year on year from Sky Sports. I do several visualizations and some regressions as well. 
26. [Sovereign Default using CUDA Julia](ipynb/pguerron-draft_defaultwithjuliacuda.ipynb)
  - *Author:* pguerron
  - *Programming Language:* Julia
  - *Published:* 2021 Oct 26
  - *Summary:* This notebook solves a sovereign default model using CUDA in Julia. 
27. [Test](ipynb/annemieke_schanze-test.ipynb)
  - *Author:* Annemieke Schanze
  - *Programming Language:* Python
  - *Published:* 2021 Sep 05
  - *Summary:* nan
28. [Simulate a Cross-section Efficiently](ipynb/julien_pascal-young_2010_quantecon.ipynb)
  - *Author:* Julien Pascal
  - *Programming Language:* Julia
  - *Published:* 2021 Jan 30
  - *Summary:* In this notebook, I describe [Young's method (2010)](https://ideas.repec.org/a/eee/dyncon/v34y2010i1p36-41.html) to simulate a large number (infinity) of individuals efficiently.
29. [Statistics](ipynb/spencer_lyon-v01_statistics_2.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Python
  - *Published:* 2021 Jan 10
  - *Summary:* This is a notebook covering basic concepts on Statistics.
30. [A brief note on Chebyshev approximation](ipynb/mohammed_aït_lahcen-chebyshev_approximation.ipynb)
  - *Author:* Mohammed Aït Lahcen
  - *Programming Language:* Python
  - *Published:* 2020 Sep 22
  - *Summary:* Examples on the use of Chebyshev polynomials for function approximation in Python.
31. [Solving the Model of Aiyagari %281994%29 with Aggregate Uncertainty](ipynb/julien_pascal-aiyagaribkm_quantecon.ipynb)
  - *Author:* Julien Pascal
  - *Programming Language:* Julia
  - *Published:* 2020 Sep 14
  - *Summary:* In this post, I solve and simulate the model of [Aiyagari (1994)](https://www.jstor.org/stable/2118417?seq=1#metadata_info_tab_contents) with aggregate uncertainty using the [BKM](https://notes.quantecon.org/submission/5ea288cb833c72001a988e4d) algorithm.
32. [Solving the Lucas Asset Pricing Model with a simple projection algorithm](ipynb/mohammed_aït_lahcen-projection_lucas_apm.ipynb)
  - *Author:* Mohammed Aït Lahcen
  - *Programming Language:* Python
  - *Published:* 2020 Aug 25
  - *Summary:* In this notebook, I use a projection method to solve a simple version of the Lucas asset pricing model.
33. [Solving the Neo-Classical Growth Model with a simple projection algorithm](ipynb/mohammed_aït_lahcen-projection_ncgm.ipynb)
  - *Author:* Mohammed Aït Lahcen
  - *Programming Language:* Python
  - *Published:* 2020 Aug 22
  - *Summary:* In this notebook, I use a simple projection algorithm to solve the discrete time neo-classical growth model.
34. [ Ramsey Plans, Time Inconsistency, Sustainable Plans](ipynb/min-calvo_julia_public.ipynb)
  - *Author:* Min
  - *Programming Language:* Julia
  - *Published:* 2020 Aug 06
  - *Summary:* This notebook is Julia version of  Ramsey Plans, Time Inconsistency, Sustainable Plans.
35. [Credible Monetary Policy in an Infinite Horizon Model: Recursive Approaches](ipynb/min-chang_julia_notebook.ipynb)
  - *Author:* Min
  - *Programming Language:* Julia
  - *Published:* 2020 Aug 06
  - *Summary:* This notebook is Julia version of Chang (1998).   - The parameter values being used below is to generate figure 25.6.2 in Ljungqvist and Sargent (4th edition).  - One can adopt the parameter values from the original paper (parameter set 3 in the code) to replicate the figure in the paper,
36. [Invariant Causal Prediction in Julia: Why Some Are Able Earn More Than 50 Thousands a Year?](ipynb/clarman_cruz-icp_salary.ipynb)
  - *Author:* Clarman Cruz
  - *Programming Language:* Julia
  - *Published:* 2020 Jul 26
  - *Summary:* Correlation does not imply causation.  Invariant Causal Prediction (ICP) is a great method allowing us to do both.  One is to find causality between the target variable and the predictors of the dataset. And two, one can predict using some of the dataset variables. The original [InvariantCausalPrediction](https://cran.rproject.org/web/packages/InvariantCausalPrediction/index.html) (ICP) and [nonlinear ICP](https://cran.r-project.org/web/packages/nonlinearICP/index.html) were implemented in R language.  This lab shows an implementation in pure [Julia](https://julialang.org) 1.4.2 of the core R language functionality of ICP.  There are numerous improvements over the original R programming.  The Julia version improves in speed, and memory usage via parallelism.  Code supportability, and usability is enhanced also.  There are new [VegaLite](https://www.queryverse.org/VegaLite.jl/stable) plots of the ICP results.  The ICP results are clearer.    The data is the [adult salary](https://www.openml.org/d/1590) set from [OpenML](https://www.openml.org/search?type=data).  The data is from 1995 but it contains data from many countries.  The lab handles a much larger data set than what the R version of ICP can process. 
37. [Aggregate uncertainty and heterogeneous agents: The BKM and GenBKM algorithms](ipynb/julien_pascal-genbkmquantecon.ipynb)
  - *Author:* Julien Pascal
  - *Programming Language:* Julia
  - *Published:* 2020 Apr 24
  - *Summary:* In this notebook, I present two algorithms to simulate **incomplete market models** with **aggregate uncertainty** that are both simple and fast:  * the BKM algorithm by [Boppart, Krusell and Mitman (2018)](https://ideas.repec.org/a/eee/dyncon/v89y2018icp68-92.html) * the GenBKM algorithm by [Reiter (2018)](https://ideas.repec.org/a/eee/dyncon/v89y2018icp93-99.html). 
38. [The linear–quadratic regulator and the Koopman operator](ipynb/julien_pascal-koopman_control_python_parti_and_ii.ipynb)
  - *Author:* Julien Pascal
  - *Programming Language:* Python
  - *Published:* 2020 Apr 17
  - *Summary:* This notebook has two goals. The first goal is to introduce to the **linear–quadratic regulator** (LQR) framework and to show how to start solving LQR problems in **Python**. The second goal is to present the **Koopman operator** and to show that it is a promising alternative to linearization.
39. [Nonlinear Invariant Causal Prediction Using Unemployment Data and Inflation Adjusted Prices from the USA Bureau of Labor](ipynb/clarman_cruz-nonlinearicp.ipynb)
  - *Author:* Clarman Cruz
  - *Programming Language:* Julia
  - *Published:* 2020 Apr 08
  - *Summary:* Correlation does not imply causation.  The purpose of this jupyter lab is to encourage conversation about how and why classical machine learning models do not handle casual interference.  The authors of [Invariant Causal Prediction for Nonlinear Models](https://arxiv.org/abs/1706.08576) (2018) not only define a sophisticated mathematical model for causal inference but also create a predictive model on top of it.  The paper's authors implemented their machine learning model in a R package named  [nonlinearICP](https://cran.r-project.org/web/packages/nonlinearICP/index.html).  The Nonlinear Invariant Causal Prediction is an interesting package.  One is able to find causality between the target variable and the predictors of the dataset. The lab shows the Nonlinear Invariant Causal Prediction package in [Julia](https://julialang.org) 1.4.0. along [VegaLite](https://www.queryverse.org/VegaLite.jl/stable) plots of the model results.  The real dataset is from [United States Bureau of Labor Statistics](https://beta.bls.gov/labs/).  The [S&P 500 Index](https://www.officialdata.org/us-economy/s-p-500-price-inflation-adjusted) prices are inflation adjusted also.  
40. [Linear Invariant Causal Prediction Using Employment Data From The Work Bank](ipynb/clarman_cruz-invariantcausalprediction.ipynb)
  - *Author:* Clarman Cruz
  - *Programming Language:* Julia
  - *Published:* 2020 Apr 01
  - *Summary:* Correlation does not imply causation.  The purpose of this jupyter lab is to encourage conversation about how and why classical machine learning models do not handle casual interference.  The authors of [Causal inference by using invariant prediction: identification and confidence intervals](https://arxiv.org/abs/1501.01332) (2016) not only define a sophisticated mathematical model for causal inference but also create a predictive model on top of it.  The paper's authors implemented their machine learning model in a R package named [InvariantCausalPrediction](https://cran.r-project.org/web/packages/InvariantCausalPrediction/index.html) (ICP). ICP is a great package allowing us to do both.  One is to find causality between the target variable and the predictors of the dataset. And two, one is able to predict using some of the dataset variables. The lab shows the Linear Invariant Causal Prediction package in [Julia](https://julialang.org) 1.4.0. along [VegaLite](https://www.queryverse.org/VegaLite.jl/stable) plots of the model results.  The real employment dataset is from [The World Bank](https://datacatalog.worldbank.org/dataset/global-jobs-indicators-database).
41. [Using Machine Learning Classifiers In Julia To Improve The Retention Rate Of Community College Students](ipynb/clarman_cruz-3models.ipynb)
  - *Author:* Clarman Cruz
  - *Programming Language:* Julia
  - *Published:* 2020 Mar 08
  - *Summary:* This jupyter lab showcases [MLJ](https://alan-turing-institute.github.io/MLJTutorials) which is a great machine learning framework.  We implement three machine learning classifiers:  [K means](https://juliastats.org/Clustering.jl/stable/kmeans.html) with [PCA](https://multivariatestatsjl.readthedocs.io/en/stable/index.html), binary [Support Vector Machine](https://www.csie.ntu.edu.tw/~cjlin/libsvm), and [Random Forrest](https://pkg.julialang.org/docs/DecisionTree/pEDeB/0.8.1) with MLJ in [Julia](https://julialang.org) 1.3.1. The three classifiers are chained because the output of one is used as part of the input of another classifier.  It is the author's unique modeling design. [VegaLite](https://www.queryverse.org/VegaLite.jl/stable) Julia package is used to graph the machine model results.   The three machine learning models help a community college to improve student experience and increase retention rate.  In this matter, the unemployment rate is reduced and state/federal tax revenue is increased as the graduates start their new jobs.  
42. [A Data Science Example In Julia Using Education Statistics From The Work Bank](ipynb/clarman_cruz-quanteconwordbankedstats.ipynb)
  - *Author:* Clarman Cruz
  - *Programming Language:* Julia
  - *Published:* 2020 Feb 16
  - *Summary:* This JupyterLab is meant as a tutorial. It showcases [Queryverse](https://www.queryverse.org),  [ARCHModels](https://s-broda.github.io/ARCHModels.jl/stable/) package, and some basic [Julia](https://julialang.org) programming. It is an end to end example using a real-world dataset from The World Bank website. We start the journey reading the statistics files, and then cleaning the data. We prepare further the data for deeper analysis. Afterwards, we graph the prepared data and create an interactive data visualizer. Then, we save our prepared data and graphs for publication, or team sharing. Lastly, we predict future time series values with a machine learning model created with the ARCHModels package.
43. [Deep learning solution method with all-in-one expectation operator](ipynb/pablo_winant-dl_notebook.ipynb)
  - *Author:* Pablo Winant
  - *Programming Language:* Python
  - *Published:* 2019 Nov 25
  - *Summary:* We show how to use TensorFlow to solve a variant of a consumption-savings model with a deep-learning approach and the All-in-One expectation operator. Companion material to the paper "Will Artificial Intelligence Replace Computational Economists Any Time Soon?" by Lilia Maliar, Serguei Maliar and Pablo Winant.
44. [Vector Autoregression](ipynb/barry_ke-vector_autoregression.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Python
  - *Published:* 2019 Oct 25
  - *Summary:* In this notebook we will run Vector Autoregression (VAR) using python packages. We will revisit the exercise from Vector Autoregression by Stock and Watson (2001).
45. [Practicing Dynare part2](ipynb/barry_ke-practicing_dynare_part_2.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Oct 25
  - *Summary:* This notebook replicates the examples in *Practicing Dynare* by Barillas, Bhandari, Colacito, Kitao, Matthes, Sargent, and Shin. 
46. [Practicing Dynare part1](ipynb/barry_ke-practicing_dynare_part1.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Oct 25
  - *Summary:* This notebook replicates the examples in *Practicing Dynare* by Barillas, Bhandari, Colacito, Kitao, Matthes, Sargent, and Shin. 
47. [Newton-Raphson method for equations with upper and lower bounds on the variable](ipynb/fedor_iskhakov-robust_newton.ipynb)
  - *Author:* Fedor Iskhakov
  - *Programming Language:* Python
  - *Published:* 2019 Aug 22
  - *Summary:* The notebook presents **bisections** and **Newton-Raphson** methods, and a **poly-algorithm combining the two**.  The latter is useful for solving equations with upper and lower bounds on the variables.   All three algorithms are implemented with a callback function parameter, which is used to build visualizations of the method iterations.  Convergence rates are also presented.  The robust Newton method is particularly useful for solving equations of the form $a \log(x) + b \log(1-x) + c = 0$, $ab<0$, which are defined on the open interval $(0,1)$ but may have the root arbitrary close to either boundary. These equations often arise in discrete choice models, for example games of two players with binary actions.
48. [Value Function Iteration](ipynb/barry_ke-value_function_iterations.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 23
  - *Summary:* In this notebook we solve a simple stochastic growth problem using value function iteration. The model is based on NYU course Quantitative Macroeconomics by Gianluca Violante
49. [Growth model with investment specific shock](ipynb/barry_ke-greenwood-hercowitz-krusell.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 20
  - *Summary:* This notebook replicates a simplified model from The role of investment-specific technological change in the business cycle (Greenwood, Hercowitz, and Krusell 1998)
50. [Applied Computational Economics and Finance--Note %28part 3%29](ipynb/barry_ke-applied_computational_economics_and_finance--part_3.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 20
  - *Summary:* This notebook contains notes and matlab implementation of some coding examples in "Applied Computational Economics and Finance" by Mario J. Miranda and Paul L. Fackler
51. [Applied Computational Economics and Finance--Note %28part 2%29](ipynb/barry_ke-applied_computational_economics_and_finance_part_2.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 20
  - *Summary:* This notebook contains notes and matlab implementation of some coding examples in "Applied Computational Economics and Finance" by Mario J. Miranda and Paul L. Fackler.
52. [Applied Computational Economics and Finance--Note %28part 1%29](ipynb/barry_ke-applied_computational_economics_and_finance.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 20
  - *Summary:* This notebook contains notes and matlab implementation of some coding examples in "Applied Computational Economics and Finance" by Mario J. Miranda and Paul L. Fackler.
53. [Applied Computational Economics and Finance--Note %28part 3%29](ipynb/barry_ke-applied_computational_economics_and_finance--part_3.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 15
  - *Summary:* This notebook contains notes and python implementation of some coding examples in "Applied Computational Economics and Finance" by Mario J. Miranda and Paul L. Fackler.
54. [Applied Computational Economics and Finance--Note %28part 2%29](ipynb/barry_ke-applied_computational_economics_and_finance_part_2.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 03
  - *Summary:* This notebook contains notes and python implementation of some coding examples in "Applied Computational Economics and Finance" by Mario J. Miranda and Paul L. Fackler. 
55. [Applied Computational Economics and Finance--Note %28part 1%29](ipynb/barry_ke-applied_computational_economics_and_finance.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 03
  - *Summary:* This notebook contains notes and python implementation of some coding examples in "Applied Computational Economics and Finance" by Mario J. Miranda and Paul L. Fackler. 
56. [Applied Computational Economics and Finance--Notes %28part 1%29](ipynb/barry_ke-applied_computational_economics_and_finance.ipynb)
  - *Author:* Barry Ke
  - *Programming Language:* Matlab
  - *Published:* 2019 Jul 03
  - *Summary:* This notebook contains notes and python implementation of some coding examples in "Applied Computational Economics and Finance" by Mario J. Miranda and Paul L. Fackler. 
57. [Python By Example](ipynb/aakash_gupta-python_by_example.ipynb)
  - *Author:* Aakash Gupta
  - *Programming Language:* Python
  - *Published:* 2019 Jul 03
  - *Summary:* nan
58. [Firm Level Innovation and CEO Compensation](ipynb/robin_li-firm_level_innovation_and_ceo_compensation.ipynb)
  - *Author:* Robin Li
  - *Programming Language:* Python
  - *Published:* 2019 May 14
  - *Summary:* Using 1992-2006 time-series data on granted patents in US and granted stock options for the Chief Executive Office (CEO) in matched US firms, this study finds a significantly positive correlation between the lagged measure of the CEO’s granted stock option value and the firm-level innovation activity (measured by number of patent granted each year). This study provides the direct empirical evidence in support of Manso (2011)’s thoery: the compensation scheme that tolerates for the early failure; rewards for the long-term success and indicates the commitment to a long-term compensation plan to CEOs, are essential in motivating innovation at the firm level.
59. [How well do FIFA ratings predict actual results?](ipynb/shahzoor-fifaproject_(1).ipynb)
  - *Author:* shahzoor
  - *Programming Language:* Python
  - *Published:* 2019 May 12
  - *Summary:* Every year, EA Sport releases a new rendition of its FIFA series, and every year I and 24 million other people flock to buy it. In the game, every player is assigned a rating between 0 and 100 based on their performances in the previous season. How specifically the ratings are calculated is somewhat opaque, but it involves some combination of performance statistics and subjective scout reports that are then reviewed by a team of editors at EA.   In this notebook, I try and examine if these ratings are any good at predicting real match outcomes. To do this, I use match data from last year's premier league season (2017/18) and ratings from this year's game, FIFA 19. That gives me a base of 380 games to work with. For every game that was played last year, I use lineup information to compute the average rating for each team and then find the difference. Then, since I know what the difference in rating was for each game and I know the result, I can effectively compute the probability distribution of outcomes for a game given the difference in average team rating. I then test how well this estimated conditional distribution does at predicting match results.  I built this for a class I took at UBC that you can find here: https://github.com/ubcecon/ECON407_2019. 
60. [pyBLP Tutorial: Post Estimation Counterfactuals](ipynb/chris_conlon-post_estimation.ipynb)
  - *Author:* Chris Conlon
  - *Programming Language:* Python
  - *Published:* 2019 May 08
  - *Summary:* pyBLP Tutorial: Part 4  Learn to calculate post-estimation quantities and counterfactual experiments. This includes elasticities, welfare estimates, HHI, and diversion ratios.  Also learn how to simulate merger effects.
61. [pyBLP Tutorial: Random Coefficients Estimation of Simultaneous Supply and Demand ](ipynb/chris_conlon-blp.ipynb)
  - *Author:* Chris Conlon
  - *Programming Language:* Python
  - *Published:* 2019 May 08
  - *Summary:* pyBLP tutorial: Part 3  Estimate the BLP 95/99 Automobile example. This problem includes simultaneous estimation of supply and demand as well as a demographic interaction between prices and income.
62. [pyBLP Tutorial: Random Coefficients Demand Estimation](ipynb/chris_conlon-nevo.ipynb)
  - *Author:* Chris Conlon
  - *Programming Language:* Python
  - *Published:* 2019 May 08
  - *Summary:* pyBLP Tutorial: Part 2  Estimate the Nevo (2000) fake cereal data example. This problem adds random coefficients, demographic interactions and fixed effects. 
63. [pyBLP Tutorial: Logit and Nested Logit Demand Estimation](ipynb/chris_conlon-logit_nested.ipynb)
  - *Author:* Chris Conlon
  - *Programming Language:* Python
  - *Published:* 2019 May 08
  - *Summary:* pyBLP tutorial: Part 1.  Learn to estimate a logit and nested logit demand model on aggregate data.
64. [Forecasting Inflation Using VAR: A Horserace Between Traditional and ML Approaches](ipynb/josephteh-final_project_teh.ipynb)
  - *Author:* josephteh
  - *Programming Language:* Python
  - *Published:* 2019 May 01
  - *Summary:* This project compares three different vector autoregressive models (VAR) in their in-sample forecasting. I utilize machine learning techniques to select features for two VARs. One of the VAR relies on a traditional technique of just choosing variables based on economic theory. Which ones do better? Find out below. 
65. [Demo of the Cost of Capital Calculator](ipynb/jason_debacker-psl_demo.ipynb)
  - *Author:* Jason DeBacker
  - *Programming Language:* Python
  - *Published:* 2019 Apr 30
  - *Summary:* This notebook provides a demo of the `Cost of Capital Calculator`, an open source tool that provides cost of capital and effective tax rates calculations on investment in the U.S.  Source code and further documentation of the model can be found in [this GitHub repo](https://github.com/PSLmodels/Cost-of-Capital-Calculator).
66. [Optimal Stopping and Linear Complementarity ](ipynb/arnav_sood-lcp_simple.ipynb)
  - *Author:* Arnav Sood
  - *Programming Language:* Julia
  - *Published:* 2019 Mar 09
  - *Summary:* Many important economic problems boil down to choosing the optimal time to stop some process.  The traditional approach to these problems is to solve a recursive, Bellman-type equation. While analytically tractable, this approach can be inconvenient for computational work.   In this notebook, we present the theory and implementation for an approach based on linear complementarity theory (LCPs).   The discretization of the problem was carried out using the QuantEcon/SimpleDifferentialOperators.jl package. A full derivation of the scheme is found in that package's documentation (link within). 
67. [Test Notebook](ipynb/aakash_gupta-chase_nelder_mead.ipynb)
  - *Author:* Aakash Gupta
  - *Programming Language:* Python
  - *Published:* 2019 Feb 19
  - *Summary:* nan
68. [test](ipynb/hobbescalvsdfsdf-kernel.ipynb)
  - *Author:* HobbesCalvsdfsdf
  - *Programming Language:* Python
  - *Published:* 2019 Feb 19
  - *Summary:* sdfsdf sdf
69. [Inflation and Unemployment in the Long Run](ipynb/mohammed_aït_lahcen-bmw_replication.ipynb)
  - *Author:* Mohammed Aït Lahcen
  - *Programming Language:* Python
  - *Published:* 2019 Jan 28
  - *Summary:* In this notebook I replicate the main results from [Berentsen, Menzio and Wright (AER, 2011)](https://www.aeaweb.org/articles?id=10.1257/aer.101.1.371) who investigate the long-run relationship between monetary policy and unemployment in the US.   The authors propose a theoretical model that combines the standard labor search model (Mortensen and Pissarides, 1994) with the standard New Monetarist model (Kyotaki and Wright, 1993; Lagos and Wright, 2005). Monetary policy affects output and employment through the real balance effect: Higher inflation increases the cost of money carried for transaction purposes which lowers consumption, reduces firms' profits from job creation and hence increases unemployment. Matching frictions in the goods market work as an amplification mechanism. Strategic complementarity between buyers' money holdings and firms' entry gives potentially rise to multiple equilibria.
70. [ Conditional Choice Probability Estimators in 5 Easy Steps!](ipynb/eric_schulman-rust.ipynb)
  - *Author:* Eric Schulman
  - *Programming Language:* Python
  - *Published:* 2019 Jan 24
  - *Summary:* The following guide demonstrates how to use conditional choice probability (CCP) estimators in Python.  These estimators are the most common way to think about how the future influences decisions in industrial organization and related economic fields.  As an example, I use the bus engine replacement problem from Rust 1987.
71. [Estimation of dynamic factor model](ipynb/shunsuke-hori-stock_watson.ipynb)
  - *Author:* Shunsuke-Hori
  - *Programming Language:* Julia
  - *Published:* 2019 Jan 22
  - *Summary:* This notebook is replicates Stock and Watson (2016, Handbook of macroeconomics) "Dynamic factor models, factor-augmented vector autoregressions, and structural vector autoregressions in macroeconomics." 
72. [Test Notebook](ipynb/aakash_gupta-chase_nelder_mead.ipynb)
  - *Author:* Aakash Gupta
  - *Programming Language:* Python
  - *Published:* 2019 Jan 08
  - *Summary:* Test summary
73. [Pikkety's Capital in the 21st Century - An Introduction](ipynb/kyle_o_shea-pikkety_chap0.ipynb)
  - *Author:* Kyle O Shea
  - *Programming Language:* Python
  - *Published:* 2018 Dec 12
  - *Summary:* Please find the live notebook, along with the next 2 chapters, here: https://kyso.io/explore/pikkety . These notebooks are completely reproducible & can be forked onto one of Kyso's Jupyterlab environments running in the cloud. 
74. [Abreu-Sannikov Algorithm for Repeated Two-player Games](ipynb/zejin_shi-as_algorithm-python.ipynb)
  - *Author:* Zejin Shi
  - *Programming Language:* Python
  - *Published:* 2018 Nov 27
  - *Summary:* This notebook demonstrates the usage of Python implementation of Abreu-Sannikov algorithm for computing the set of payoff pairs of all pure-strategy subgame-perfect equilibria with public randomization for any repeated two-player games with perfect monitoring and discounting.
75. [Examples of parallel value function iteration in Julia](ipynb/andrew_owens-parallel_vfi_examples.ipynb)
  - *Author:* Andrew Owens
  - *Programming Language:* Julia
  - *Published:* 2018 Nov 21
  - *Summary:* I show how to do value function iteration for a simple savings problem in Julia, and how to use multithreading or distributed memory processing to gain speedups from parallel computing.
76. [Continuous Sequential Importance Resampling for Stochastic Volatility Models](ipynb/davide_viviano-seqmc_notebook.ipynb)
  - *Author:* davide viviano
  - *Programming Language:* R
  - *Published:* 2018 Nov 05
  - *Summary:* The notebook implements Continuous SIR in R with a wrapper function in C. Code available also on Github. The project has been developed by  Hans-Peter Hollwirth, Robert T. Lange and Davide Viviano. 
77. [DiscreteDP: Implementation Details](ipynb/daisuke_oyama-ddp_theory_jl.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Julia
  - *Published:* 2018 Oct 30
  - *Summary:* This notebook describes the implementation details of the `DiscreteDP` type and its methods in `QuantEcon.jl`.
78. [DiscreteDP: Implementation Details](ipynb/daisuke_oyama-ddp_theory_py.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Python
  - *Published:* 2018 Oct 30
  - *Summary:* This notebook describes the implementation details of the `DiscreteDP` class in `QuantEcon.py`.
79. [DiscreteDP: Getting Started with a Simple Example](ipynb/daisuke_oyama-ddp_intro_py.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Python
  - *Published:* 2018 Oct 30
  - *Summary:* This notebook demonstrates via a simple example how to use the `DiscreteDP` module in `QuantEcon.py`.
80. [Introduction to DBnomics in Python](ipynb/christophe_benz-index.ipynb)
  - *Author:* Christophe Benz
  - *Programming Language:* Python
  - *Published:* 2018 Oct 26
  - *Summary:* How to use DBnomics to fetch data from Python
81. [Stocks, Significance Testing & p-Hacking: how volatile is volatile?](ipynb/patrick_david-kernel.ipynb)
  - *Author:* Patrick David
  - *Programming Language:* Python
  - *Published:* 2018 Oct 21
  - *Summary:*  **Over the past 32 years, October has been the most volatile month on average for the S&P500 and December the least, in this article we will use simulation to assess the statistical significance of this observation and to what extent this observation could occur by chance.** 
82. [YouTube Trending Videos Analysis](ipynb/ammar_alyousfi-youtube_trending_videos_analysis.ipynb)
  - *Author:* Ammar Alyousfi
  - *Programming Language:* Python
  - *Published:* 2018 Oct 20
  - *Summary:* Analysis of more than 40,000 YouTube trending videos. Python is used with some packages like Pandas and Matplotlib to analyze a dataset that was collected over 205 days. For each of those days, the dataset contains data about the trending videos of that day. It contains data about more than 40,000 trending videos.   
83. [Testing Financial Strategies with R: Stock Picking](ipynb/alejandro_jiménez-kernel.ipynb)
  - *Author:* Alejandro Jiménez
  - *Programming Language:* R
  - *Published:* 2018 Oct 19
  - *Summary:* One of the most common financial advices that you can hear in every Christmas meal is that you should be saving a fixed amount of money each month. This statement usually arises cocky arguments about the hottest blue-chips to invest on.  In this Notebook I'll explore a systematic methodology to test different stock-picking strategies
84. [A Problem that Stumped Milton Friedman](ipynb/chase_coleman-wald_friedman.ipynb)
  - *Author:* Chase Coleman
  - *Programming Language:* Python
  - *Published:* 2018 Oct 19
  - *Summary:* This notebook describes a problem faced by the Statistical Research Group during World War 2 that stumped Milton Friedman and which Abraham Wald solved by inventing Sequential Analysis.
85. [Solving Krusell-Smith model](ipynb/shunsuke-hori-krusellsmith.ipynb)
  - *Author:* Shunsuke-Hori
  - *Programming Language:* Julia
  - *Published:* 2018 Oct 04
  - *Summary:* This notebook solves the Krusell-Smith model to replicate Maliar, Lilia, Serguei Maliar, and Fernando Valli (2010, JEDC)
86. [No, Python is Not Too Slow for Computational Economics](ipynb/john_stachurski-python_vs_julia.ipynb)
  - *Author:* John Stachurski
  - *Programming Language:* Python
  - *Published:* 2018 Sep 28
  - *Summary:* A response to a paper by Boragan Aruoba and Jesus Fernández-Villaverde.
87. [Hamilton filter](ipynb/shunsuke-hori-hamilton_filter.ipynb)
  - *Author:* Shunsuke-Hori
  - *Programming Language:* Julia
  - *Published:* 2018 Sep 20
  - *Summary:* This notebook introduces `hamilton_filter` and `hp_filter` in QuantEcon.jl and describes "Why you should never use the Hodrick-Prescott filter" with some examples in Hamilton (2017).
88. [Tools for Game Theory in GameTheory.jl](ipynb/daisuke_oyama-game_theory_jl.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Julia
  - *Published:* 2018 Aug 08
  - *Summary:* This notebook demonstrates the functionalities of `GameTheory.jl`.
89. [Tools for Game Theory in QuantEcon.py](ipynb/daisuke_oyama-game_theory_py.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Python
  - *Published:* 2018 Aug 08
  - *Summary:* This notebook demonstrates the functionalities of the `game_theory` module in `QuantEcon.py`.
90. [helloworld](ipynb/linda-warm-up_exercise.ipynb)
  - *Author:* linda
  - *Programming Language:* Python
  - *Published:* 2018 Aug 01
  - *Summary:* hello world 2
91. [sdfsdfsdfsdf](ipynb/linda-warm-up_exercise.ipynb)
  - *Author:* linda
  - *Programming Language:* Python
  - *Published:* 2018 Aug 01
  - *Summary:* Hello World
92. [7 Solution Methods for Neoclassical Growth Model %28Matlab%29](ipynb/chase_coleman-growthmodelsolutionsmethods_matlab.ipynb)
  - *Author:* Chase Coleman
  - *Programming Language:* Other
  - *Published:* 2018 Jul 30
  - *Summary:* This file is part of a computational appendix that accompanies the paper.  > MATLAB, Python, Julia: What to Choose in Economics?  Coleman, Lyon, Maliar, and Maliar (2017)  
93. [7 Solution Methods for Neoclassical Growth Model %28Julia%29](ipynb/chase_coleman-growthmodelsolutionmethods_jl.ipynb)
  - *Author:* Chase Coleman
  - *Programming Language:* Julia
  - *Published:* 2018 Jul 30
  - *Summary:* This file is part of a computational appendix that accompanies the paper.  > MATLAB, Python, Julia: What to Choose in Economics?  Coleman, Lyon, Maliar, and Maliar (2017)
94. [7 Solution Methods for Neoclassical Growth Model %28Python%29](ipynb/chase_coleman-growthmodelsolutionmethods_py.ipynb)
  - *Author:* Chase Coleman
  - *Programming Language:* Python
  - *Published:* 2018 Jul 30
  - *Summary:* This file is part of a computational appendix that accompanies the paper.  > MATLAB, Python, Julia: What to Choose in Economics?  Coleman, Lyon, Maliar, and Maliar (2017)
95. [Very Simple Markov Perfect Industry Dynamics in Python](ipynb/john_stachurski-industry_dynamics.ipynb)
  - *Author:* John Stachurski
  - *Programming Language:* Python
  - *Published:* 2018 Jul 24
  - *Summary:* An implementation of the ECMA 2018 paper "Very Simple Markov Perfect Industry Dynamics" by Abbring et al, written in Python.  Computation of the equilibrium is accelerated using Numba.
96. [Krusell Smith](ipynb/mario_silva-krusell_smith.ipynb)
  - *Author:* Mario Silva
  - *Programming Language:* Python
  - *Published:* 2018 Jul 23
  - *Summary:* Solution of Krusell-Smith model via time iteration on household slide and convergence of perceived law of motion of the capital stock with actual law of motion. Adapted algorithm by Lilia Maliar, Serguei Maliar, and Fernando Valli  
97. [A stylized New Keynesian Model](ipynb/spencer_lyon-model.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Other
  - *Published:* 2018 Jul 22
  - *Summary:* # A stylized New Keynesian Model  This notebook is part of a computational appendix that accompanies the paper.  > MATLAB, Python, Julia: What to Choose in Economics?  > > Coleman, Lyon, Maliar, and Maliar (2017)  This contains the model.
98. [Solving a New Keynesian model with Python](ipynb/spencer_lyon-python.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Python
  - *Published:* 2018 Jul 22
  - *Summary:* # Solving a New Keynesian model with Python  This notebook is part of a computational appendix that accompanies the paper.  > MATLAB, Python, Julia: What to Choose in Economics?  >> Coleman, Lyon, Maliar, and Maliar (2017)
99. [Solving a New Keynesian model with Matlab](ipynb/spencer_lyon-matlab.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Other
  - *Published:* 2018 Jul 22
  - *Summary:* # Solving a New Keynesian model with Matlab  This notebook is part of a computational appendix that accompanies the paper.  > MATLAB, Python, Julia: What to Choose in Economics?  >> Coleman, Lyon, Maliar, and Maliar (2017)
100. [ Solving a New Keynesian model with Julia](ipynb/spencer_lyon-julia.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Julia
  - *Published:* 2018 Jul 22
  - *Summary:* # Solving a New Keynesian model with Julia  This file is part of a computational appendix that accompanies the paper.  > MATLAB, Python, Julia: What to Choose in Economics? > > Coleman, Lyon, Maliar, and Maliar (2017)
101. [Autor Dorn and Hanson 2013 replication](ipynb/spencer_lyon-czone_analysis_ipw_final.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Python
  - *Published:* 2018 Jul 22
  - *Summary:* In this notebook I use the tools in the PyData ecosystem to replicate the main regressions and figures from  Autor, D. H., Dorn, D., & Hanson, G. H. (2013).
102. [Gaussian process regression an applications to economics](ipynb/spencer_lyon-asgp_py.ipynb)
  - *Author:* Spencer Lyon
  - *Programming Language:* Python
  - *Published:* 2018 Jul 22
  - *Summary:* In this notebook we describe the basic mathematical theory behind Gaussian process regression and replicate a few of the examples from Machine learning for high-dimensional dynamic stochastic economies" by Scheidegger/Bilionis 2017
103. [A quick test](ipynb/natasha-linearisation.ipynb)
  - *Author:* Natasha
  - *Programming Language:* Python
  - *Published:* 2018 Jul 16
  - *Summary:* nan
104. [Business Cycle Moment Calculator](ipynb/kerk_phillips-moment_calculator.ipynb)
  - *Author:* Kerk Phillips
  - *Programming Language:* Python
  - *Published:* 2018 Jul 06
  - *Summary:* This notebook shows how to easily calculate business cycle moments from a data set.  Includes examples of four common filters. The code in this notebook uses Python 3.6.
105. [The Auberbach-Kotlikoff Fixed-Point Solver with Adaptive Dampening](ipynb/kerk_phillips-ak_fixed_point_solver.ipynb)
  - *Author:* Kerk Phillips
  - *Programming Language:* Python
  - *Published:* 2018 Jul 06
  - *Summary:* This notebook provides examples of how to use the A-K fixed point solver with an adaptive dampener. There are two examples for steady states: 1) Solow model and 2) OLG model. The code in this notebook uses Python 3.6.
106. [Calculating Euler Errors](ipynb/kerk_phillips-euler_errors.ipynb)
  - *Author:* Kerk Phillips
  - *Programming Language:* Python
  - *Published:* 2018 Jul 06
  - *Summary:* This notebook provides an example of how to calculate Euler errors for a simple DSGE model.  There are two examples: 1) linearization and 2) value-function iteration. The code in this notebook uses Python 3.6.
107. [Solving a Stochastic OLG Model using DSGE Tools](ipynb/kerk_phillips-olg_dsge.ipynb)
  - *Author:* Kerk Phillips
  - *Programming Language:* Python
  - *Published:* 2018 Jul 06
  - *Summary:* This notebook provides an example of how to solve and simulate a simple stochastic OLG model using the same linearization tools as are commonly used to solve DSGE models.. The code in this notebook uses Python 3.6.
108. [Solving a Simple DSGE Model using Linearization](ipynb/kerk_phillips-dsge_linapp.ipynb)
  - *Author:* Kerk Phillips
  - *Programming Language:* Python
  - *Published:* 2018 Jul 06
  - *Summary:* This notebook provides an example of how to solve and simulating a simple DSGE model using linearization about the model's steady state. The code in this notebook uses Python 3.6.
109. [Solving a Simple DSGE Model using Value-Function Iteration](ipynb/kerk_phillips-dsge_grid.ipynb)
  - *Author:* Kerk Phillips
  - *Programming Language:* Python
  - *Published:* 2018 Jul 06
  - *Summary:* This notebook provides an example of how to solve and simulating a simple DSGE model using value-function iteration on a full cartesian grid.  The code in this notebook uses Python 3.6.
110. [Simulated Method of Moments %28SMM%29 Estimation](ipynb/richard_w_evans-smmest.ipynb)
  - *Author:* Richard W Evans
  - *Programming Language:* Python
  - *Published:* 2018 Jul 05
  - *Summary:* This notebook provides a characterization of the simulated method of moments (SMM) approach to parameter estimation in the general setting of a nonlinear functions and non-Gaussian errors. Code uses Python 3.6.
111. [Generalized Method of Moments %28GMM%29 Estimation](ipynb/richard_w_evans-gmmest.ipynb)
  - *Author:* Richard W Evans
  - *Programming Language:* Python
  - *Published:* 2018 Jul 03
  - *Summary:* This notebook provides a characterization of the generalized method of moments (GMM) approach to parameter estimation in the general setting of a nonlinear functions and non-Gaussian errors. Code uses Python 3.6.
112. [Generalized Method of Moments Estimation](ipynb/richard_w_evans-gmmest.ipynb)
  - *Author:* Richard W Evans
  - *Programming Language:* Python
  - *Published:* 2018 Jul 03
  - *Summary:* This notebook provides a characterization of the generalized method of moments (GMM) approach to parameter estimation in the general setting of a nonlinear functions and non-Gaussian errors. Code uses Python 3.6.
113. [Maximum Likelihood Estimation](ipynb/richard_w_evans-mlest.ipynb)
  - *Author:* Richard W Evans
  - *Programming Language:* Python
  - *Published:* 2018 Jul 03
  - *Summary:* This notebook provides a characterization of maximum likelihood approach to parameter estimation in the general setting of a nonlinear functions and non-Gaussian errors. Code uses Python 3.6 to explore the theory and computation behind MLE.
114. [Maximum Likelihood Estimation, structural estimation](ipynb/richard_w_evans-mlest.ipynb)
  - *Author:* Richard W Evans
  - *Programming Language:* Python
  - *Published:* 2018 Jul 03
  - *Summary:* This notebook provides a characterization of maximum likelihood approach to parameter estimation in the general setting of a nonlinear functions and non-Gaussian errors. We use Python 3.6 to explore the theory and computation behind MLE.
115. [A Critical Analysis of Engel and Rogers %281996%29 Using Python Visualizations](ipynb/quentin_batista-a_critical_analysis_of_engel_and_rogers_(1996)_using_python_visualizations.ipynb)
  - *Author:* Quentin Batista
  - *Programming Language:* Python
  - *Published:* 2018 Jul 02
  - *Summary:* This notebook investigates the existence of the border effect between U.S. and Canadian prices initially established by Engel and Rogers (1996) by examining the adequacy of their model and data using insightful visualizations.
116. [Cryptocurrency Returns Are Heavy-Tailed](ipynb/quentin_batista-cryptocurrency_returns_are_heavy-tailed.ipynb)
  - *Author:* Quentin Batista
  - *Programming Language:* Python
  - *Published:* 2018 Jul 02
  - *Summary:* This notebook analyzes the empirical distribution of cryptocurrency returns and establishes that they follow a heavy-tailed distribution.
117. [Finite Markov Chains: Examples](ipynb/daisuke_oyama-markov_chain_ex01_py.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Python
  - *Published:* 2018 Jun 29
  - *Summary:* This notebook demonstrates how to analyze finite-state Markov chains with the `MarkovChain` class from `QuantEcon.py`.
118. [DiscreteDP Example: Automobile Replacement](ipynb/daisuke_oyama-ddp_ex_rust96_py.ipynb)
  - *Author:* Daisuke Oyama
  - *Programming Language:* Python
  - *Published:* 2018 Jun 29
  - *Summary:* This notebook demonstrates how to use the `DiscreteDP` class from `QuantEcon.py` to solve the finite-state version of the automobile replacement problem as considered in Rust (1996), "Numerical Dynamic Programming in Economics," Handbook of
119. [Advanced Data Analysis in Python: Computing the ProductSpace](ipynb/matt_mckay-advanced-data-analysis-python-hidalgo2007.ipynb)
  - *Author:* Matt McKay
  - *Programming Language:* Python
  - *Published:* 2018 Jun 27
  - *Summary:* In this notebook I demonstrate a few of the ``Python`` ecosystem tools that enable **research** in areas that can be difficult to do using traditional tools such as ``Stata`` that are typically fit-for-purpose statistical tools. The agility of a full programming language environment allows for a high degree of flexibility and the Python ecosystem provides a vast toolkit to remain productive.   This notebook is motivated by computing the Product Space network, which uses conditional probability of co-export as a measure of product similarity. It is often best visualised as a network.
120. [High Performance Python with Numba](ipynb/john_stachurski-vectorization_numba.ipynb)
  - *Author:* John Stachurski
  - *Programming Language:* Python
  - *Published:* 2018 Jun 22
  - *Summary:* This notebook gives a quick introduction to Numba and JIT compilation within Python's scientific computing environment.  With Numba, basic mathematical operations in Python can run as fast as Fortran.