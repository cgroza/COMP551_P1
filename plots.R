library(tidyverse)
library(ggplot2)

gdweigths <- read_csv("weights.tsv", col_names = T) %>% gather(key = "Term", "Value")

weight.plot <- ggplot(gdweigths) + geom_boxplot(aes(x = Term, y = Value)) + geom_point(aes(x=Term, y=Value)) +
    ggtitle("Gradient descent stability") + labs(x = "Linear regression coefficient", y = "Value at convergence")
ggsave("WeightStability.pdf", weight.plot)

