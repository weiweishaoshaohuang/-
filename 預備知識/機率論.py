import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
# print(multinomial.Multinomial(1,fair_probs).sample()) # 從六種可能隨機抽樣其中一個
# print(multinomial.Multinomial(10,fair_probs).sample()) # 六種可能隨機抽樣十個
counts = multinomial.Multinomial(1000,fair_probs).sample()
print(counts / 1000) # 用頻率當作機率




