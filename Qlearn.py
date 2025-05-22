import numpy as np
import pandas as pd
import time
import sys
import os

import matplotlib.pyplot as plt

np.random.seed(200)  # reproducible

N_STATES = 4   # the length of the 1 dimensional world
M_STATES = 4
ACTIONS = ['[0,1]','[0,-1]','[1,0]', '[-1,0]']     # available actions
# EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 100   # maximum episodes
FRESH_TIME = 0.2 # fresh time for one move

orrM = np.array([
    [0, 5, 3, 0],
    [4, -2, 1, -3],
    [0, -4, 6, 5],
    [3, -2, -1, 20]
])
# M = orrM-6
# M[0,0]=0

global visited # 记录是否访问过
visited = np.zeros(N_STATES*M_STATES)

'''
所构造的q表的大小是：行数是财产的探索者总共所处的位置数，列数就是动作的数量
'''
def build_q_table(total_states, actions):
    table = pd.DataFrame(
        np.random.rand(total_states, len(actions)),     # q_table initial values
        columns=actions,    # actions's name
    )
    return table

def islegal(state,visited,state_actions,options,ago):
    # 可选项为：wall    Not 不允许撞墙 Try 允许撞墙（会有惩罚）
	# 	  back   Not 不允许返回上一步 Never 不允许走之前走过的路 Allow 啥都可以

    for s in state_actions.index:
        xs = state%4
        ys = state//4
        list_s = eval(s)
        nowvec = [a + b for a, b in zip([xs, ys], list_s)]

        wall = options['wall']
        back = options['back']

        if wall=='Not' and (nowvec[0] >= M.shape[0] or nowvec[0] < 0 or nowvec[1] >= M.shape[1] or nowvec[1] < 0):# 不允许撞墙
            if s in state_actions.index:
                state_actions.drop(s, inplace=True)
        elif wall=='Try':# 允许撞墙（会有惩罚）
            state_actions=state_actions
        else:
            Warning('undefined wall')
        
        nowindex = nowvec[0]+nowvec[1]*4
        if back=='Not' and nowindex==ago:# 不允许返回上一步
            if s in state_actions.index:
                state_actions.drop(s, inplace=True)
        elif back=='Never' and (nowindex>0 and nowindex<N_STATES*M_STATES-1) and visited[nowindex] != 0:# 不允许走之前走过的路
            if s in state_actions.index:
                state_actions.drop(s, inplace=True)
        elif back=='Allow':# 啥都可以
            state_actions=state_actions
        else:
            Warning('undefined back')

    # print('-------')
    # print('state')
    # print(state)
    # print('ago')
    # print(ago)
    # print('action')
    # print(state_actions)
    # print('-------')
    return state_actions

def choose_action(state,q_table,epsilon,visited,options, ago = None):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :] #将现在agent观测到的状态所对应的q值取出来
    state_actions = islegal(state,visited,state_actions,options,ago)
    if state_actions.empty:
        return 'NAN'
    way = options['way']
    if (np.random.uniform() > epsilon):  #当生撑随机数大于EPSILON，就随机选择状态state所对应的动作
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        if way == 'Q':
            action_name = state_actions.idxmax()    # 选择状态state所对应的使q值最大的动作
        elif way == 'S':
            action_name = np.random.choice(state_actions.index) # 随机选一个
        elif way == 'R':
            state_actions_ = state_actions - np.min(state_actions)+1
            probabilities = state_actions_ / state_actions_.sum() # Q值加权随机
            action_name = np.random.choice(state_actions.index, p=probabilities)
        else:
            Warning('undefined way')

    return action_name

def get_env_feedback(S, A):
    # 这个函数就是根据当前的状态和动作，返回下一个状态和奖励
    # 选择动作之后，还要根据现在的状态和动作获得下一个状态，并且返回奖励，这个奖励是环境给出的，用来评价当前动作的好坏。
    
    #这里设置的是，状态解码为xy坐标，利用提前存储的矩阵得到奖励函数
    x = S%4
    y = S//4
    list_A = eval(A)
    vec = [a + b for a, b in zip([x, y], list_A)]

    if vec[0] >= M.shape[0] or vec[0] < 0 or vec[1] >= M.shape[1] or vec[1] < 0:
        return S,-10
    else:
        nextplace = vec[0]+vec[1]*4
        return nextplace,M[tuple(np.flip(vec))]

def update_env(S, episode, step_counter, getEnd,options):
    if options['Video'] == 'Not':
        return #这个函数暂时不使用
    elif options['Video'] == 'Yes':
    # 更新环境的函数，比如向右移动之后，o表示的agent就距离宝藏进了一步，将agent随处的位置实时打印出来
        env_list = ['-']*(N_STATES*M_STATES-1)+['T']   # '---------T' our environment
        if S == 15:
            interaction = 'Attend End %s Episode %s: total_steps = %s' % (getEnd,episode+1, step_counter)
            sys.stdout.write(
                    '\033[F\033[K' * 4 +      # 上移4行并逐行清除
                    str(interaction) + '\n\n\n\n'    # 输出新内容（自动换行到下一行）
                )
            sys.stdout.flush()
            time.sleep(1)
        else:
            env_list[S] = 'o'
            for i in range(N_STATES):
                if i == N_STATES-1:
                    continue
                env_list.insert((N_STATES-i-1)*M_STATES,'\n')
            interaction = ''.join(env_list)
            sys.stdout.write(
                    '\033[F\033[K' * 4 +      # 上移4行并逐行清除
                    str(interaction) + '\n'    # 输出新内容（自动换行到下一行）
                )
            sys.stdout.flush()
            time.sleep(FRESH_TIME)
    else:
        Warning('undefined draw')
    
def init_reward_matrix(subtract=6):
    global M,Subtract
    M = orrM - subtract
    Subtract = subtract
    # M[0,0] = 0

def rl(options, progress_callback=None):
    # 初始化奖励矩阵
    subtract = options.get('subtract', 3)
    init_reward_matrix(subtract)
    
    # 使用传入的MAX_EPISODES参数，如果没有则使用默认值100
    MAX_EPISODES = options.get('MAX_EPISODES', 10)
    
    if options['Video'] == 'Yes':
        env_list = ['\n']*(N_STATES+1)
        interaction = ''.join(env_list)
        print(interaction, end='')
    total_steps = []
    EPISODE = 0
    
    q_table = build_q_table(N_STATES*M_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):  # 使用动态设置的MAX_EPISODES
        # 计算并发送进度
        if progress_callback:
            progress = int(100 * (episode + 1) / MAX_EPISODES)
            progress_callback(progress)
            
        step_counter = 0
        S = 0
        S_ = 0
        S__ = 0
        getEnd = True
        visited = np.zeros(N_STATES*M_STATES)
        is_terminated = False
        update_env(S, episode, step_counter, getEnd, options)
        
        while not is_terminated:
            visited[S] += 1
            epsilon = EPISODE + (episode / MAX_EPISODES) * (1-EPISODE)*1.5
            A = choose_action(S, q_table,epsilon,visited,options=options,ago = S__)#agent根据当前的状态选择动作
            
            S_, R = get_env_feedback(S, A)  # 上一步已经获得了s对应的动作a，接着我们要获得下一个时间步的状态
            A_ = choose_action(S_, q_table,epsilon,visited,options=options,ago = S)
            if  A_ == 'NAN':
                S_ = 15
                R = -10
                getEnd = False
            q_predict = q_table.loc[S, A]
            if S_ != 15:#要判断一下，下一个时间步的是不是已经取得宝藏了，如果不是，可以按照公式进行更新
                func = options['func']
                if func == 'QL':
                    q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
                elif func == 'SA':
                    q_target = R + GAMMA * q_table.loc[S_, A_]   # next state is not terminal
                elif func == 'EXP':
                    q_target = R + GAMMA * q_table.iloc[S_, :].mean()   # next state is not terminal
                else:
                    Warning('undefined func')
            else:#如果已经得到了宝藏，得到的下一个状态不在q表中，q_target的计算也不同。
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S__ = S
            S = S_  # move to next state

            update_env(S, episode, step_counter+1,getEnd,options)
            step_counter += 1
            if step_counter>1000:
                Warning('too long')
                break
        total_steps.append(step_counter)
        if any(visited>1):
            Warning('why twice')
    
    # 添加可视化代码
    os.makedirs('./images', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.plot(range(1, len(total_steps)+1), total_steps)
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')
    plt.title('Learning Progress')
    
    options_str = '  '.join([f'{k}: {v}' for k, v in options.items() if k != 'iterations'and k!='warning_callback'])
    plt.text(0.5, 0.9, f"Options:  {options_str}", 
            ha='center', va='center', transform=plt.gca().transAxes,
            fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(f'images/learning_progress_{timestamp}.png')
    # plt.show()
    # plt.close()
    return q_table

def extract_path(q_table,options0):
    options = options0.copy()
    options['func'] = 'QL'
    options['way'] = 'Q'
    path = []
    step_counter = 0
    epsilon = 1
    S = 0
    S_ = 0
    S__ = 0
    getEnd = True
    visited = np.zeros(N_STATES*M_STATES)
    is_terminated = False
    while not is_terminated:
        
        path.append(S)
        visited[S] += 1
        A = choose_action(S, q_table,epsilon,visited,options=options,ago = S__)
        
        S_, R = get_env_feedback(S, A)
        # print(f'现在状态是S{S},将要通过A{A}到达S_{S_}')
        if choose_action(S_, q_table,epsilon,visited,options=options,ago = S) == 'NAN':
            S_ = 15
            R = -10
            getEnd = False
        q_predict = q_table.loc[S, A]
        if S_ != 15:
            q_target = R + GAMMA * q_table.iloc[S_, :].max()  # 添加这行赋值
        else:
            path.append(15)
            q_target = R
            is_terminated = True

        # q_table.loc[S, A] += ALPHA * (q_target - q_predict)
        S__ = S
        S = S_

        step_counter += 1
        if step_counter>1000:
            warning_msg = 'The final result is just like a died, but it is not the end.'
            if 'warning_callback' in options:
                options['warning_callback'](warning_msg)
            else:
                Warning(warning_msg)
            break
    return path

def draw(q_table,options):
    # 生成路径数据
    path = extract_path(q_table,options0=options)
    print(f'最终路径为{path}')
    # print(path)
    # 创建绘图
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.grid(True)
    
    # 新增分数矩阵显示（在单元格中心显示M矩阵的值）
    for i in range(4):
        for j in range(4):
            ax.text(j , 3 - i , f'{orrM[i, j]:.0f}',
                    fontsize=12, color='black', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    # 先计算完整路径的分数范围
    final_score = Subtract
    all_scores = []
    for state in path:
        y = state // 4
        x = state % 4
        final_score += M[tuple([y, x])]
        all_scores.append(final_score)
    
    min_score = min(all_scores)
    max_score = max(all_scores)
    score_range = max_score - min_score if max_score != min_score else 1  # 防止除以0

    # 绘制路径箭头
    for i, state in enumerate(path[:-1]):
        x = state % 4
        y = 3 - state // 4
        
        next_state = path[i+1]
        dx = (next_state % 4) - x
        dy = (3 - next_state // 4) - y

        # 使用完整路径的分数范围计算颜色
        norm_score = (all_scores[i] - min_score) / score_range
        # print(norm_score)
        arrow_color = plt.cm.plasma(norm_score)
        
        ax.arrow(x, y, dx*0.3, dy*0.3, 
                head_width=0.15, 
                head_length=0.15, 
                fc=arrow_color,
                ec=arrow_color)
        
        # 移动文本标注到循环内（保持原有位置）
        ax.text(x + dx*0.5, y + dy*0.5, f'score:{all_scores[i]:.1f}', 
               fontsize=12, color='red',ha='center', va='center')

    # 统一在循环外添加colorbar ↓↓↓
    sm = plt.cm.ScalarMappable(cmap='plasma', 
                             norm=plt.Normalize(vmin=min(all_scores), vmax=max(all_scores)))  # 修正变量名
    plt.colorbar(sm, ax=ax, label='Cumulative Score')
    
    # 标记起点和终点
    ax.plot(0, 3, 'go', markersize=15)
    ax.plot(3, 0, 'r*', markersize=20)
    plt.title(f'Optimal Path Visualization with score {final_score}')

    os.makedirs('./images', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.title(f'Optimal Path Visualization with score {final_score}')
    print(f'最终得分为{final_score}')
    options_str = '  '.join([f'{k}: {v}' for k, v in options.items() if k != 'iterations'and k!='warning_callback'])
    plt.text(0.5, 1.1, f"{options_str}", 
            ha='center', va='center', transform=plt.gca().transAxes,
            fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
    plt.savefig(f'images/path_visualization_{timestamp}.png')
    # plt.close()
    # plt.show()

if __name__ == "__main__":
    # 可选项为：wall    Not 不允许撞墙 Try 允许撞墙（会有惩罚）
	# 	  back   Not 不允许返回上一步 Never 不允许走之前走过的路 Allow 啥都可以
    #     Q选择状态state所对应的使q值最大的动作   S随机选一个   R Q值加权随机

    options = {'func':'SA','way': 'Q','wall':'Not','back':'Not','Video':'Not'}
    q_table = rl(options)
    print('\r\nQ-table:\n')
    print(q_table)

    draw(q_table,options)