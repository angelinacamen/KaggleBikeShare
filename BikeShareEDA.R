bike <- vroom("~/Downloads/KaggleBikeShare/bike-sharing-demand/train.csv")
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)

weather_bp <- ggplot(bike, aes(x = weather, y = count, fill = weather)) +
  geom_bar(stat = "identity") +
  labs(title = "Weather Barplot",
       x = "Weather",
       y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

humidity_plot <- ggplot(data=bike, aes(x=humidity, y=count)) +
geom_point() +
geom_smooth(se=FALSE)

season_bp <- ggplot(bike, aes(x=factor(season))) +
  geom_bar(fill="steelblue") +
  theme_minimal() + 
  labs(x = "Season", y = "count")

tempseason_boxplot <- ggplot(bike, aes(x = factor(season), y = temp, fill=factor(season))) +
  geom_boxplot(outlier.color="black", outlier.shape=16, outlier.size=3) +
  theme_minimal() + 
  labs(x = "season", y = "temp", fill = "season")

tempseason_boxplot + season_bp
(tempseason_boxplot + season_bp)/(weather_bp + humidity_plot)

