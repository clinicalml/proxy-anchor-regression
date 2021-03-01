library(tidyverse)
library(dplyr)

# Import and convert data
df = read_csv("experiment1-data.csv")
method.names = c("ar" = "AR(A)", "cross" = "xPAR(W, Z)", "par" = "PAR(W)", "ols" = "OLS")
df$method <- factor(df$method, levels=names(method.names), labels=method.names)

# Dataframe containing theoretical values
df.theo <- df %>% subset(n == "theo") %>% select(method, value, x)

# Create data frame containing finite sample values
df <- df %>%
  subset(n != "theo" & method %in% c("xPAR(W, Z)", "PAR(W)")) %>%
  transform(n = factor(as.integer(n)))
levels(df$n) <- paste0("$n = ", levels(df$n), "$")

# Plot
p <- ggplot(df, aes(x=x, y=value, colour=method, fill=method)) +
  geom_line(data=df.theo, aes(linetype=method, colour=method), size=0.8) +
  stat_summary(geom="line", fun=median, alpha=1, size=1)+
  stat_summary(geom="ribbon", fun.data=median_hilow, fun.args=list(conf.int=0.5), size=0.1, alpha=0.1, show.legend = F) +
  coord_cartesian(ylim=c(0.65, 2.5)) +
  scale_linetype_manual(values=c("22", "26", "22", "22"), breaks = c("PAR(W)"), labels="", name="Population")+
  scale_color_brewer(palette = "Dark2", name="Method", breaks = levels(df.theo$method), labels = method.names) +
  scale_fill_brewer(palette = "Dark2", name="Method", labels = method.names) +
  labs(y="MSPE under $do(A:=\\nu)$",
       x="Signal-to-variance ratio") +
  scale_x_continuous(breaks = c(0:4)/4, labels=c("0\\%", "25\\%", "50\\%", "75\\%", "100\\%")) +
  theme_bw(base_size=9) +
  theme(legend.position = "bottom",
        legend.spacing.y = unit(0.05, 'cm'),
        legend.spacing.x = unit(0.05, 'cm'),
        legend.margin=margin(c(-10,0,-5, 0)),
        legend.key.width = unit(0.3,"cm"),
        legend.title = element_text(size = 9),
        plot.margin = margin(0, 0, 0, 0)) +
  guides(color=guide_legend(order=2, title=NULL),
         linetype=guide_legend(order=1, override.aes = list(lty = "22"))) +
  facet_wrap(~n)
print(p)
