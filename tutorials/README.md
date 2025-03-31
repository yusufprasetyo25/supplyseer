Tutorials will soon be filled with in-depth guides on how to use the modules and the library.

# Economic Order Quantity (EOQ)

## Overview

EOQ is the optimal solution for simple inventory model with defined `demand`, `order_cost`, and `holding_cost`.
* Example code can be seen in `economic-order-quantity/eoq.ipynb`

On the other hand, there are some cases when we cannot rely solely on the empirical values of those parameters. Thus, we introduce the Bayesian EOQ.

Main idea: we assume that the parameters are probability distributions, hence the EOQ is also probability distribution.

To that end, most of the work in Bayesian EOQ is on preparing the parameters distribution, rather than the formula for EOQ.
* Example code can be seen in `economic-order-quantity/bayesian_eoq.ipynb`
