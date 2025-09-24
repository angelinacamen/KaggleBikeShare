library(tidymodels)
install.packages("glmnet")
install.packages("vroom")

# Load the package
library(vroom)



bikeTrain <- vroom("~/Downloads/KaggleBikeShare/bike-sharing-demand/train.csv") 
bikeTest <- vroom("~/Downloads/KaggleBikeShare/bike-sharing-demand/test.csv") #drop casual and registered variables

#Data Cleaning
bikeTrain <- bikeTrain %>% select(-c(casual, registered)) %>% mutate(count = log(count))


#Feature Engineering using a recipe
my_recipe <- recipe(count ~., data = bikeTrain) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features=c("hour")) %>%
  step_mutate(season=factor(season)) %>%
  step_zv(all_predictors())
prepped_recipe <- prep(my_recipe) 
bake(prepped_recipe, new_data=bikeTest)

## Define a Recipe as before
bike_recipe <- recipe(count ~., data = bikeTrain) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_time(datetime, features=c("hour")) %>%
  step_mutate(season=factor(season)) %>%
  step_zv(all_predictors())

#Linear Regression Workflow
## Define a Model
lin_model <- linear_reg() %>%
set_engine("lm") %>%
set_mode("regression")

## Combine into a Workflow and fit
bike_workflow <- workflow() %>%
add_recipe(bike_recipe) %>%
add_model(lin_model) %>%
fit(data=bikeTrain)

## Run all the steps on test data
lin_preds <- predict(bike_workflow, new_data = bikeTest)

##Back transforming log(count) prediction
lin_preds <- lin_preds %>%
  mutate(.pred = exp(.pred))

##Printing first five rows of baked dataset
prepped_recipe <- prep(bike_recipe)
baked_train <- bake(prepped_recipe, new_data = bikeTest)
head(baked_train, 5)

##Prepare file for Kaggle Submission
kaggle_submission_wf <- lin_preds %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission_wf, file="./BakedData.csv", delim=",")

##Penalized Regression Model
penalty_value <- 1   
mixture_value <- 0.01
preg_model <- linear_reg(penalty=penalty_value, mixture=mixture_value) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R


my_recipe <- recipe(count ~., data=bikeTrain) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime, features=c("hour")) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_rm(datetime) %>%
  step_zv(all_predictors()) %>%
  step_lincomb(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) #Make mean 0, sd=1

prepped_recipe <- prep(my_recipe)


preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(lin_model) %>%
  fit(data=bikeTrain)


preds <- predict(preg_wf, new_data=bikeTest)
preds <- preds %>%
  mutate(.pred = exp(.pred))

kaggle_submission <- preds %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./PenalizedRegression2.csv", delim=",")



##Tuning and Fitting a Penalized Regression Model
# Penalized regression model
preg_model <- linear_reg(penalty=tune(), mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 6) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(bikeTrain, v = 6, repeats=1)

# Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
          grid=grid_of_tuning_params,
          metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Plot Results 
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric="rmse")

# Finalize the Workflow & fit it
final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=bikeTrain)

## Predict
final_preds <- final_wf %>%
  predict(new_data = bikeTest)

##Prepare for Kaggle Submission
kaggle_submission <- final_preds %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./TuningandFitting1.csv", delim=",")


##Regression Trees
install.packages("rpart")
library(tidymodels)

my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model
  set_engine("rpart") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
regtree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(
  tree_depth(),
  cost_complexity(),
  min_n(),
  levels = 3
) 

## Split data for CV
folds <- vfold_cv(bikeTrain, v = 3, repeats = 1)

# Run the CV
CV_results <- regtree_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(rmse, mae)
  )

# Plot results (use tree_depth or min_n instead of penalty/mixture)
collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(x = tree_depth, y = mean, color = factor(min_n))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

# Finalize the workflow & fit it
final_wf <- regtree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bikeTrain)


## Predict
final_preds <- final_wf %>%
  predict(new_data = bikeTest)

##Prepare for Kaggle Submission
kaggle_submission <- final_preds %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./RegressionTree.csv", delim=",")


##Random Forests
install.packages("ranger")
library(ranger)
install.packages("rpart")
library(tidymodels)
## maxnumx is how many columns are in the baked data set
my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=1000) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
randfor_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(mtry(range=c(1, ncol(baked_train))),
                                      min_n(), 
                                      levels = 5) 

## Split data for CV
folds <- vfold_cv(bikeTrain, v = 6, repeats = 1)

# Run the CV
CV_results <- randfor_wf %>%
  tune_grid(
    resamples = folds,
    grid = grid_of_tuning_params,
    metrics = metric_set(rmse, mae)
  )

# Plot results (use tree_depth or min_n instead of penalty/mixture)
collect_metrics(CV_results) %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(x = tree_depth, y = mean, color = factor(min_n))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "rmse")

# Finalize the workflow & fit it
final_wf <- randfor_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = bikeTrain)


## Predict
final_preds <- final_wf %>%
  predict(new_data = bikeTest)
finals_preds <- final_preds %>%
  mutate(.pred = exp(.pred))

##Prepare for Kaggle Submission
kaggle_submission <- final_preds %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./RandomForrests.csv", delim=",")


##Linear Predictions
linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(formula=count~.-datetime, data = bikeTrain)

## Generate Predictions Using Linear Model
bike_predictions <- predict(linear_model,
                            new_data=bikeTest) # Use fit to predict
bike_predictions ## Look at the output 


## Format the Predictions for Submission to Kaggle1
kaggle_submission <- bike_predictions %>%
bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

