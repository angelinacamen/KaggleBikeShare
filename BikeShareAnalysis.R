library(tidymodels)
install.packages("glmnet")


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
penalty_value <- 5      
mixture_value <- 0.05 
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

kaggle_submission <- bike_predictions %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and prediction variables
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write out the file
vroom_write(x=kaggle_submission, file="./PenalizedRegression1.csv", delim=",")


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

