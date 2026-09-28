# Diffusion Policy
 告诉你往哪里走
 >multimodal action distribution  

 加噪怎么加，直接加吗，加的方式有影响吗，也影响了去噪哈哈

 obseravtion_step action_step  execute_step
 ```text
at time step t the policy takes the latest To  steps of observation data Ot  as input and
predicts Tp  steps of actions, of which Ta  steps of actions are
executed on the robot without re-planning . Here, we define To
as the observation horizon, Tp  as the action prediction horizon
and  Ta   as  the  action  execution  horizon .  
```
 预测层：
 >CNN-based
poorly: low-frequency
 >Transformer-based 这里竟然在action这里用的是causal mask