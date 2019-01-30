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


gd.errors <- read_csv("gd_error.csv", col_names = T)
ls.errors <- tibble(R2 = 0.8374430425869106, MSE = 0.9954868683378355)

error.plot <- ggplot(gd.errors) + geom_point(aes(x = R2, y = MSE)) + geom_point(data = ls.errors, aes(x=R2, y = MSE), color = "blue") +
    ggtitle("Validation R-squared vs Mean Square Error") + labs(x = "R-squared", y = "MSE") + scale_y_continuous(limits = (0.80, 1.20)) + + scale_x_continuous(limits = (0.60, 1))

ggsave("ValidationError.pdf", error.plot)


word.trend <- tibble(word.features = c(0, 30, 60, 90, 160), Validating = c(0.9954868683378355,0.9981891717698733, 0.9998974894145352, 1.010936093561357, 1.0169063196640469),
                     Testing = c(1.2929916316350558, 1.3235819430470093, 1.3568025010225244,
                              1.3394241454727076, 1.3631631524252839)) %>% gather(key = Dataset , value = "MSE", -word.features)
word.plot <- ggplot(word.trend) + geom_line(aes(x=word.features, y=MSE, color = Dataset)) + geom_point(aes(x=word.features, y=MSE, color = Dataset)) + ggtitle("Number of text features vs MSE") +
    labs(x = "Number of text features", y = "MSE") #+ scale_y_continuous(limits = c(0.80, 1.20))

ggsave("WordFeaturesMSE.pdf", word.plot)
