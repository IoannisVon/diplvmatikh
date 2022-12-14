import torch
from collections import deque
from gym.wrappers import Monitor
from os import mkdir
from os.path import join
from torch.nn.functional import softmax
from .constants import PRETRAINED_MODELS
from .helpers import initialize_model
from .train_information import TrainInformation
from .wrappers import wrap_environment
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

def infer(model, env):
    episode_reward = 0.0
    state = torch.from_numpy(env.reset())
    hx = torch.zeros((1, 512), dtype=torch.float)
    cx = torch.zeros((1, 512), dtype=torch.float)

    while True:
        logit, value, hx, cx = model(state, hx, cx)
        policy = softmax(logit, dim=1)
        action = torch.argmax(policy).item()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = torch.from_numpy(next_state)
        hx = hx.detach()
        cx = cx.detach()
        if done:
            return


def record_best_run(model, args, episode):
    # Simple function to always record videos
    def record(episode_id):
        return True

    env = wrap_environment(args.environment, args.action_space)
    env = Monitor(env, 'recordings/run%s' % episode, force=True,
                  video_callable=record)
    # Update the framerate to 20 frames per second for a more naturally-paced
    # playback.
    env.metadata['video.frames_per_second'] = 20.0
    infer(model, env)

def diagrammatxt(a,b,c): 
  filename = f"A3C_.txt" 
  with open(filename,'a+') as f:
      f.write(str(a * 10)+','+str(b)+','+str(c)+'\n')
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
  plt.plot(x,y, color='indigo')
  plt.title('A3C') # subplot 211 title
  plt.ylabel('Reward')
  plt.xlabel('Episodes')
  plt.savefig("A3C.png")

def complete_episode(environment, info, episode_reward, episode, stats, model,
                     flag, args, action_list):
    best = info.best_reward
    new_best = info.update_rewards(episode_reward)
    save_model = False
    save_flag = False
    if episode_reward > best:
        print('New high score of %s! Saving model' % round(episode_reward, 3))
        save_model = True
    
    if flag:
        save_flag = info.store_actions(action_list)
        save_model = True
    if save_model:
        record_best_run(model, args, episode)
        torch.save(model.state_dict(),
                   join(PRETRAINED_MODELS, '%s.dat' % environment))
    if save_flag:
        mkdir('saved_models/run%s' % episode)
        torch.save(model.state_dict(),
                   join('saved_models/run%s' % episode,
                        '%s.dat' % environment))
    print('Episode %s - Reward: %s, Best: %s, Average: %s'
          % (episode,
             round(episode_reward, 3),
             round(info.best_reward, 3),
             round(info.average, 3)))



def test_loop(env, model, global_model, actions, state, done, args, info,
              episode_reward, hx, cx, action_list):
    flag = False
    if done:
        model.load_state_dict(global_model.state_dict())
    with torch.no_grad():
        if done:
            hx = torch.zeros((1, 512), dtype=torch.float)
            cx = torch.zeros((1, 512), dtype=torch.float)
        else:
            hx = hx.detach()
            cx = cx.detach()
    logit, value, hx, cx = model(state, hx, cx)
    policy = softmax(logit, dim=1)
    action = torch.argmax(policy).item()
    next_state, reward, done, stats = env.step(action)
    if args.render:
        env.render()
    episode_reward += reward
    actions.append(action)
    action_list.append(action)
    if stats['flag_get']:
        print('Reached the flag!')
        #flag = True #test an sunexizei meta to flag 
    if done or actions.count(actions[0]) == actions.maxlen:
        diagrammatxt(episode_reward, stats["time"], stats["flag_get"])
        done = True
        info.update_index()
        complete_episode(args.environment, info, episode_reward, info.index,
                         stats, model, flag, args, action_list)
        episode_reward = 0.0
        actions.clear()
        action_list.clear()
        next_state = env.reset()
    state = torch.from_numpy(next_state)
    return model, hx, cx, state, done, info, episode_reward


def test(env, global_model, args):
    torch.manual_seed(123 + args.num_processes)
    info = TrainInformation()
    env = wrap_environment(args.environment, args.action_space)
    model = initialize_model(env, args.environment, args.transfer)
    model.eval()

    state = torch.from_numpy(env.reset())
    done = True
    episode_reward = 0.0
    hx = None
    cx = None
    actions = deque(maxlen=args.max_actions)
    action_list = []

    while True:
        loop_outputs = test_loop(env, model, global_model, actions, state,
                                 done, args, info, episode_reward, hx, cx,
                                 action_list)
        model, hx, cx, state, done, info, episode_reward = loop_outputs
