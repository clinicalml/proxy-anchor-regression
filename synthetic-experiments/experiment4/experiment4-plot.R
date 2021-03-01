library(ggplot2)
library(dplyr)

# Load data
df = read.csv2("experiment4-data.csv", sep=",", stringsAsFactors = T)
df$mse <- as.numeric(as.character(df$mse))

# Name methods
method.names = c(
  "ols" = "OLS",
  "ar" = "AR(A)",
  "tar" = "TAR(A)"
)
df$method <- factor(df$method, levels=c('tar', 'ar', 'ols'))

# Label settings
setup.labeller = c(
  "correct_shift" = "Anticipated shift occuring",
  "incorrect_shift" = "Anticipated shift not occuring"
)

# Make plots
p <- ggplot(df, aes(x=mse, color=method, fill=method)) + 
  geom_histogram(aes(y = ..density..), position="identity", bins = 100, alpha=0.8) + 
  facet_wrap(~setup, ncol=1, labeller = as_labeller(setup.labeller)) + 
  scale_fill_brewer(palette = "Dark2", name="Method", labels = method.names) +
  scale_color_brewer(palette = "Dark2", name="Method", labels = method.names) +
  theme_bw(base_size=9) + 
  theme(legend.title=element_blank(),
        panel.grid = element_blank(),
        legend.key.size = unit(0.8,"line"),
        legend.spacing.y = unit(0.1, 'cm'),
        plot.margin = margin(0,0,0,0)) +
  labs(x="MSPE", y=NULL)
print(p)
