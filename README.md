<div align="center>
<p>
    <a target="_blank">
      <img width="100%" src="https://github.com/jakorostami/supplyseer/blob/feature/development-over-time/assets/supplyseerfront.png" alt="SupplySeer Vision banner"></a>
  </p>

# <div align="center"> SupplySeer </div>
Welcome to version 0.11!

`supplyseer` is a Python library focused on providing the tools and methods for real-world Supply Chain & Logistics challenges. <br>
<br>
You'll find Bayesian Economic Order Quantity (dynamical stochastic EOQ), Probabilistic Bayesian Networks, Neural Networks, <br>
Principal Component Analaysis, time series models like ARIMA, and evaluation metrics for models and for information content. <br>
<br>
Supplyseer provides Permutation Complexity as a metric for time series analysis but also Manipulability Index and Hurst Exponent and many more.
<br>
<br>

## <div align="center"> Models </div>
Below are some models listed

| Model | Use case |
| --- | --- |
| [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) | Time Series |
| [Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network) | Uncertainty Modeling, Prediction, and Causal Inference |
| [Bayesian EOQ](https://en.wikipedia.org/wiki/Economic_order_quantity) | Economic Order Quantity with distributions instead of fixed values |
| [Neural Network](https://en.wikipedia.org/wiki/Neural_network_(machine_learning)) | Machine Learning Modeling | 
| [Mixture Density Network](https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/) | Probabilistic Machine Learning Modeling with multi-modal data |
| [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) | Embeddings in Machine Learning (dimensionality reduction) |

<br>

## <div align="center"> Tools & Metrics </div>

| Name | Use case | 
| --- | --- |
| [Time Upsampling](https://en.wikipedia.org/wiki/Upsampling) | Good when you have missing dates between samples |
| [Taken's Embeddings](https://en.wikipedia.org/wiki/Takens%27s_theorem) | Extract the dynamics of a time series/signal |
| Economic Order Quantity | This is the basic function of EOQ that returns a value while the Bayesian EOQ is a dynamic model |
| [Manipulability Index](https://iaeme.com/MasterAdmin/Journal_uploads/IJMET/VOLUME_6_ISSUE_6/IJMET_06_06_002.pdf) | Another way of measuring volatility and stability of a time series |
| [Hurst Exponent R/S](https://en.wikipedia.org/wiki/Hurst_exponent) | Measure long-term memory or autocorrelation in a time series |
| [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) | Measures the unpredictability or randomnesss |
| [Permutation Entropy](https://materias.df.uba.ar/dnla2019c1/files/2019/03/permutation_entropy.pdf) | Quantifies the diversity of patterns in the ordinal structure of a time series. It is the first output of `permutation_complexity()`|
| [Statistical Complexity](https://arxiv.org/pdf/1009.1498) | Measures the structural complexity of a system. It combines entropy with disequilibrium (a measure of structure). It is the second output of `permutation_complexity()`|
</div>
