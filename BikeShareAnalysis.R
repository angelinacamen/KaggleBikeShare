library(tidymodels)

bikeTrain <- vroom("~/Downloads/KaggleBikeShare/bike-sharing-demand/train.csv") 
bikeTest <- vroom("~/Downloads/KaggleBikeShare/bike-sharing-demand/test.csv") #drop casual and registered variables

bikeTrain <- bikeTrain %>% select(-c(casual, registered))


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

