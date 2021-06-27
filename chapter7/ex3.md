# Is it possible to speed up training of bagging ensemble by distributing it across multiple servers? What about pasting ensembles, boosting ensembles, Random forrest or stacking ensembles?

## Bagging ensamble
Yes, the different submodels are trained completely independent.
As this is with replacement, even the selection of the samples can happen in parallel.

## Pasting ensembles
Yes, the different submodels are trained completely independent.
As there is not replacement, you can only use every sample once.
So you do need to divide up the samples first, before parallizing the training.
Not sure if this makes sence perfomance wise.

## Boosting
No, the training of the different models happens in a certain sequence. 
As the models after the first model need the residual of the previous one.

## Random forres
Yes, the different submodels are trained completely independent.

## Stacking
Yes partially, all models except the blender are trained independently, so it's only the blender that needs to happen in one place.
