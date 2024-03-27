
## data
* * *
24.3.1 - 24.3.22

## result
* * *
Giving up due to lack of logical basis for changing ingredients set to ingredients sequence.
Fourier augmentation is possible only by changing ingredients set to ingredients sequence.

## dataset
* * *
ing_mlm_data (in ubuntu)
-> title [SEP] ingredients [SEP] tags

## recipetrend
* * *
task : recipe recommendation
backbone : recipebuild
method : recipe sequence + fourier augmentation + contrastive learning + trend sampling
reference : 
1) ?
2) Contrastive Learning with Frequency-Domain Interest Trends for Sequential Recommendation
3) Capturing Popularity Trends: A Simplistic Non-Personalized Approach for Enhanced Item Recommendation 


## stack
* * *
pytorch
numpy

## Details
* * * 
Use transformer package
 
## wandb
* * * 
recipetrend_test -> recipebuild(baseline) loss + recipetrend((1,1,1) setting) loss