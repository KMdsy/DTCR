# Read me file

[Chinese version](./readme_cn.md)

Tensorflow implementation of paper 'Learning Representations for Time Series Clustering' (NIPS 2019 accept paper).
**This code is not the official version.**

Details
>
> Ma, Q., Zheng, J., Li, S., & Cottrell, G. W. (2019). Learning representations for time series clustering. 
> In Advances in neural information processing systems (pp. 3781-3791).

Bibtex
```
@inproceedings{ma2019learning,
  title={Learning representations for time series clustering},
  author={Ma, Qianli and Zheng, Jiawei and Li, Sen and Cottrell, Gary W},
  booktitle={Advances in neural information processing systems},
  pages={3781--3791},
  year={2019}
}
```



## Some results

RI (Rand Index) is employed as performance (same as the paper). 
I used [this version](https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation) of the RI implementation since there is no official implementation method in sklearn package.

I run each experiment runs 5 times and report means and stand deviations.
The `best` column represents the best performance in all the experiments.
The `paper` column lists the RI reported by the paper.


### Configs
Config1：encoder_hidden_units = [100, 50, 50], lambda = 1,

Config2：encoder_hidden_units = [100, 50, 50], lambda = 0.1,

Config3：encoder_hidden_units = [100, 50, 50], lambda = 0.01,

Config4：encoder_hidden_units = [100, 50, 50], lambda = 0.001,

Config5：encoder_hidden_units = [50, 30, 30], lambda = 1,

Config6：encoder_hidden_units = [50, 30, 30], lambda = 0.1,

Config7：encoder_hidden_units = [50, 30, 30], lambda = 0.01,

Config8：encoder_hidden_units = [50, 30, 30], lambda = 0.001.

### Results
Data preprocessing method: N/A

| Dataset               	| config1           	| config2           	| config3           	| config4           	| config5           	| config6           	| config7           	| config8           	| best    	| paper           	|
|-----------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|---------	|-----------------	|
| ArrowHead             	| 0.63103 ± 0.04962 	| 0.64632 ± 0.02547 	| 0.6402 ± 0.04928  	| 0.66869 ± 0.02821 	| 0.6562 ± 0.0493   	| **0.67823 ± 0.04251** 	| 0.64906 ± 0.05363 	| 0.6529 ± 0.03482  	| 0.74023 	| 0.6868 ± 0.0026 	|
| Beef                  	| 0.7669 ± 0.02558  	| 0.76644 ± 0.02347 	| 0.77471 ± 0.02122 	| **0.77793 ± 0.02044** 	| 0.7577 ± 0.00926  	| 0.74897 ± 0.00958 	| 0.75954 ± 0.01854 	| 0.76 ± 0.01204    	| 0.81609 	| 0.8046 ± 0.0018 	|
| BeetleFly             	| 0.60526 ± 0       	| 0.61684 ± 0.02316 	| **0.68737 ± 0.10056** 	| 0.60526 ± 0       	| 0.60526 ± 0       	| 0.60526 ± 0       	| 0.63053 ± 0.05053 	| 0.67158 ± 0.08497 	| 0.81052 	| 0.9000 ± 0.0001 	|
| BirdChicken           	| 0.66211 ± 0.07688 	| 0.58211 ± 0.08346 	| **0.74737 ± 0.03158** 	| 0.67632 ± 0.10017 	| 0.54737 ± 0.06781 	| 0.57789 ± 0.10082 	| 0.59684 ± 0.06451 	| 0.61474 ± 0.11087 	| 0.81053 	| 0.8105 ± 0.0033 	|
| Car                   	| 0.64667 ± 0.03581 	| 0.68316 ± 0.03617 	| 0.71537 ± 0.01632 	| **0.71797 ± 0.01905** 	| 0.6304 ± 0.02426  	| 0.65695 ± 0.01937 	| 0.69153 ± 0.018   	| 0.71073 ± 0.03539 	| 0.77401 	| 0.75.1 ± 0.0022 	|
| ChlorineConcentration 	| 0.52175 ± 0.01628 	| 0.51549 ± 0.01654 	| 0.5276 ± 0.01301  	| 0.53374 ± 0.00277 	| 0.5222 ± 0.01634  	| 0.51528 ± 0.01587 	| 0.52575 ± 0.0123  	| **0.53555 ± 0.00072** 	| 0.53659 	| 0.5357 ± 0.0011 	|
| Coffee                	| 0.68624 ± 0.17581 	| 0.65132 ± 0.12575 	| 0.78995 ± 0.10818 	| **0.85397 ± 0.18698** 	| 0.58942 ± 0.11309 	| 0.60741 ± 0.04073 	| 0.79365 ± 0.1563  	| 0.82381 ± 0.16011 	| 1       	| 0.9286 ± 0.0016 	|



Data preprocessing method: Normalized

| Dataset               	| config1           	| config2           	| config3           	| config4           	| config5           	| config6           	| config7           	| config8           	| best     	| paper           	|
|-----------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|-------------------	|----------	|-----------------	|
| ArrowHead             	| 0.61923 ± 0.05194 	| 0.61398 ± 0.04337 	| 0.65328 ± 0.02648 	| 0.66475 ± 0.03845 	| 0.6055 ± 0.03643  	| 0.65639 ± 0.03132 	| **0.67137 ± 0.02044** 	| 0.66328 ± 0.03323 	| 0.71278  	| 0.6868 ± 0.0026 	|
| Beef                  	| 0.70713 ± 0.00892 	| 0.70575 ± 0.00497 	| 0.71667 ± 0.01364 	| 0.72337 ± 0.00217 	| 0.70851 ± 0.01202 	| 0.72138 ± 0.01457 	| 0.71552 ± 0.01791 	| **0.72414 ± 0.00291** 	| 0.74483  	| 0.8046 ± 0.0018 	|
| BeetleFly             	| 0.71842 ± 0.16428 	| 0.67105 ± 0.08392 	| 0.73509 ± 0.06021 	| 0.74211 ± 0.09676 	| 0.62842 ± 0.02836 	| **0.75789 ± 0.10771** 	| 0.66421 ± 0.11789 	| 0.74211 ± 0.10458 	| 1        	| 0.9000 ± 0.0001 	|
| BirdChicken           	| 0.53579 ± 0.05702 	| 0.58596 ± 0.05674 	| 0.64511 ± 0.09452 	| **0.67193 ± 0.08203** 	| 0.50877 ± 0.02796 	| 0.56632 ± 0.09342 	| 0.65 ± 0.10556    	| 0.64868 ± 0.02507 	| 0.81053  	| 0.8105 ± 0.0033 	|
| Car                   	| 0.70927 ± 0.01742 	| 0.71119 ± 0.02797 	| **0.72249 ± 0.02928** 	| 0.7096 ± 0.02585  	| 0.69085 ± 0.01935 	| 0.70395 ± 0.01768 	| 0.71073 ± 0.03181 	| 0.71921 ± 0.01226 	| 0.77288  	| 0.75.1 ± 0.0022 	|
| ChlorineConcentration 	| 0.50288 ± 0.00019 	| 0.50821 ± 0.01159 	| 0.51451 ± 0.0144  	| 0.53447 ± 0.00096 	| 0.50255 ± 0.00008 	| 0.5083 ± 0.01156  	| 0.51469 ± 0.01472 	| **0.53519 ± 0.00106** 	| 0.053889 	| 0.5357 ± 0.0011 	|



# Requirements

Tensorflow>=1.13.2
