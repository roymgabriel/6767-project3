# ISYE/MATH 6767 Project 3
## Statistical Arbitrage Trading Strategy


Statistical arbitrage (a.k.a stat-arb) strategies are commonly employed in quantitative asset management
practices. Pairs trading is a well-known example of stat-arb strategies. The goal of this project is to implement
a stat-arb strategy as proposed in [[Avellaneda and Lee 2010]](doc/mathematics-09-00179.pdf) and examine the strategy performance
over a universe of 40 crypto currencies (or, tokens). The hourly price data of more than 120 tokens are taken
from FTX over the time period of 2021-2-19 to 2022-9-26 and saved in file “coins all prices.csv”. Among
these tokens, the 40 with largest market capitalizations in each hour based on their respective market prices
are recorded in file “coins universe 150K 40.csv”. THIS IS AN UPDATE do this