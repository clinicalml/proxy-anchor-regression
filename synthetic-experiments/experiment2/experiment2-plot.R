library(tidyverse)
library(dplyr)

# Import and convert data
df = read_csv("experiment2-data.csv")
# The assumed signal-to-variance ratio
svr = 0.4

# Transform names
var.names <- c("actual"="PAR($W$) MSPE, true", "belief" = "PAR($W$) MSPE, est.", "ols" = "OLS MSPE")
df$variable <- factor(df$variable, levels = names(var.names), labels=var.names)

# Plot
p <- ggplot(df, aes(x=x, y=value, colour=variable, fill=variable, lty="Default")) +
  stat_summary(geom="line", fun=median, alpha=1, size=0.8, show.legend=T) +
  stat_summary(data=df, geom="ribbon", fun.data=median_hilow, fun.args=list(conf.int=0.5), size=0.00, alpha=0.1, show.legend=F)+
  geom_vline(mapping=aes(xintercept=svr, lty="Assumed SVR"), size=0.6, show.legend=F) +
  labs(y="MSPE", x="True signal-to-variance ratio") +
  scale_x_continuous(breaks = (1:4)/4, labels = c("25\\%", "50\\%", "75\\%", "100\\%")) +
  scale_linetype_manual(values = c("22", "solid"), breaks="Assumed SVR", name=NULL) +
  scale_color_brewer(palette="Dark2", aesthetics = c("color", "fill"), name=NULL) +
  theme_bw(base_size=9) + 
  theme(legend.margin = margin(0,0,0,0), 
        legend.spacing.y = unit(3, 'pt'),
        plot.margin = margin(0, 0, 0, 0))
print(p)
