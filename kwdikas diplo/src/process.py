import torch
from src.env import create_train_env
from src.model import PPO
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

def diagrammatxt(a,b,c): 
  filename = f"PPO_.txt" 
  with open(filename,'a+') as f:
      f.write(str(a)+','+str(b)+','+str(c)+'\n')
  with open(filename,'r') as f:
    a=f.read()

  #z=[]
  x=[]
  y=[]
  #print('1. a=\n',a)

  a=a.split('\n')
  a.pop()
  #print('2. a=',a)
  temp=0
  for i in a:
    temp += 1
    #print('3. a=',i)
    i=i.split(',')
    #print('4. a=',i)
    x.append(int(temp))
    y.append(float(i[0]))
  mpl.style.use('seaborn')
  plt.plot(x,y, color='darkred')
  plt.title('PPO') # subplot 211 title
  plt.ylabel('Reward')
  plt.xlabel('Episodes')
  plt.savefig("PPO_.png")



def eval(opt, global_model, num_states, num_actions):
    torch.manual_seed(123)
    if opt.action_type == "right":
        actions = RIGHT_ONLY
    elif opt.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(opt.world, opt.stage, actions)
    local_model = PPO(num_states, num_actions)
    if torch.cuda.is_available():
        local_model.cuda()
    local_model.eval()
    state = torch.from_numpy(env.reset())
    if torch.cuda.is_available():
        state = state.cuda()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    episode_reward=0.0
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
        episode_reward += reward
        # Uncomment following lines if you want to save model whenever level is completed
        if info["flag_get"]:
        # if random.randint(0, 10)%2 == 0:
            # print("Finished")
            torch.save(local_model.state_dict(),
                       "{}/ppo_super_mario_bros_{}_{}_{}".format(opt.saved_path, opt.world, opt.stage, curr_step))
            # return

        # env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            diagrammatxt(episode_reward,info["time"],info["flag_get"])
            print(episode_reward)
            episode_reward = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
        if torch.cuda.is_available():
            state = state.cuda()
