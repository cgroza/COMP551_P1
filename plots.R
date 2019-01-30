library(tidyverse)
library(ggplot2)

gdweigths <- read_csv("weights.tsv", col_names = T) %>% gather(key = "Term", "Value")

lsweights <- tibble(Term =c("W1","W2","W3","W4","W5","W6"), Value = c(-1.09098000e+00,  4.05789421e-01, -2.28999224e-01,  1.19484466e-03, -8.56546449e-04,  7.91201104e-01))

weight.plot <- ggplot(gdweigths) + geom_boxplot(aes(x = Term, y = Value)) + geom_jitter(aes(x=Term, y=Value)) +
    geom_point(data=lsweights, aes(x=Term, y=Value), color = "blue") +
    ggtitle("Gradient descent stability") + labs(x = "Linear regression coefficient", y = "Value at convergence")
ggsave("WeightStability.pdf", weight.plot)

least.squres.times <- read_csv("least_squares_time.csv", col_names = T)
gd.times <- read_csv("DEFgradient_descent.csv", col_names = T)

times <- tibble(Closed=least.squres.times$LEAST_SQUARES, Gradient = gd.times$GRADIENT_DESCENT_DEF) %>% gather(key = "Method", "Time")
time.plot <- ggplot(times) + geom_boxplot(aes(x = Method, y = Time)) + geom_jitter(aes(x=Method, y=Time)) +
    ggtitle("Training time comparison") + labs(x = "Method", y = "Training time (s)")
ggsave("TrainingTimes.pdf", time.plot)
