library(tidyverse)
library(ggplot2)

gdweigths <- read_csv("weights.tsv", col_names = T) %>% gather(key = "Term", "Value")

weight.plot <- ggplot(gdweigths) + geom_boxplot(aes(x = Term, y = Value)) + geom_jitter(aes(x=Term, y=Value)) +
    ggtitle("Gradient descent stability") + labs(x = "Linear regression coefficient", y = "Value at convergence")
ggsave("WeightStability.pdf", weight.plot)

least.squres.times <- read_csv("least_squares_time.csv", col_names = T)
gd.times <- read_csv("DEFgradient_descent.csv", col_names = T)

times <- tibble(Closed=least.squres.times$LEAST_SQUARES, Gradient = gd.times$GRADIENT_DESCENT_DEF) %>% gather(key = "Method", "Time")
time.plot <- ggplot(times) + geom_boxplot(aes(x = Method, y = Time)) + geom_jitter(aes(x=Method, y=Time)) +
    ggtitle("Training time comparison") + labs(x = "Method", y = "Training time (s)")
ggsave("TrainingTimes.pdf", time.plot)
