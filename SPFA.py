from io import BytesIO  # 添加这行
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

def visualize_spfa(matrix, path, orrM=None):
    """
    可视化SPFA算法找到的最短路径
    
    参数:
        matrix: 转换后的矩阵
        path: 找到的最短路径
        orrM: 原始矩阵(可选)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建颜色映射
    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
    
    # 绘制矩阵和colorbar
    im = ax.imshow(matrix, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, label='Value')
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # 显示数值 - 使用原始矩阵或转换后的矩阵
    display_matrix = orrM if orrM is not None else matrix
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            ax.text(j, i, f"{display_matrix[i][j]:.1f}", 
                   ha="center", va="center", color="w", fontsize=12)
    
    # 标记起点和终点
    start = path[0]
    end = path[-1]
    ax.add_patch(patches.Rectangle((start[1]-0.5, start[0]-0.5), 1, 1, 
                                  linewidth=2, edgecolor='r', facecolor='none'))
    ax.add_patch(patches.Rectangle((end[1]-0.5, end[0]-0.5), 1, 1, 
                                  linewidth=2, edgecolor='g', facecolor='none'))
    
    # 绘制路径
    for i in range(len(path)-1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        ax.arrow(y1, x1, y2-y1, x2-x1, 
                head_width=0.3, head_length=0.3, 
                fc='red', ec='red', linewidth=2)
    
    plt.title("SPFA Algorithm - Shortest Path")
    plt.show()

def visualize_negative_cycle(matrix, cycle, orrM=None):
    """
    可视化负环路径
    
    参数:
        matrix: 转换后的矩阵
        cycle: 负环路径
        orrM: 原始矩阵(可选)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 创建颜色映射
    cmap = plt.cm.viridis
    norm = colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
    
    # 绘制矩阵和colorbar
    im = ax.imshow(matrix, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, label='Value')
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # 显示数值 - 使用原始矩阵或转换后的矩阵
    display_matrix = orrM if orrM is not None else matrix
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            ax.text(j, i, f"{display_matrix[i][j]:.1f}", 
                   ha="center", va="center", color="w", fontsize=12)
    
    # 绘制负环路径
    for i in range(len(cycle)-1):
        x1, y1 = cycle[i]
        x2, y2 = cycle[i+1]
        ax.arrow(y1, x1, y2-y1, x2-x1, 
                head_width=0.3, head_length=0.3, 
                fc='red', ec='red', linewidth=2)
    
    # 标记负环起点
    ax.add_patch(patches.Rectangle((cycle[0][1]-0.5, cycle[0][0]-0.5), 1, 1, 
                                  linewidth=2, edgecolor='yellow', facecolor='none'))
    
    plt.title("SPFA Algorithm - Negative Cycle Detected")
    plt.show()

def spfa_shortest_path(matrix, start, end, orrM=None, prevent_back_and_forth=False, gui_mode=False):
    """
    使用SPFA算法寻找矩阵中从起点到终点的最短路径
    
    参数:
        matrix: 二维矩阵，每个元素代表移动到此位置的距离
        start: 起点坐标 (x, y)
        end: 终点坐标 (x, y)
        orrM: 原始矩阵(可选)
        prevent_back_and_forth: 是否禁止反复横跳
    """
    rows = len(matrix)
    if rows == 0:
        return (float('inf'), []) if not gui_mode else (float('inf'), [], None)
    cols = len(matrix[0])
    
    # 将二维坐标转换为一维索引
    def coord_to_index(x, y):
        return x * cols + y
    
    # 将一维索引转换为二维坐标
    def index_to_coord(index):
        return (index // cols, index % cols)
    
    size = rows * cols
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    # 初始化距离和前驱节点
    dist = [float('inf')] * size
    prev = [-1] * size
    in_queue = [False] * size
    count = [0] * size  # 记录每个节点的入队次数
    
    # 起点初始化
    start_idx = coord_to_index(start[0], start[1])
    end_idx = coord_to_index(end[0], end[1])
    dist[start_idx] = 0
    queue = deque([start_idx])
    in_queue[start_idx] = True
    count[start_idx] += 1
    
    # 初始化前驱方向记录
    prev_direction = [None] * size
    
    # SPFA主循环
    while queue:
        u = queue.popleft()
        in_queue[u] = False
        
        # 如果当前节点是终点，则不再探索其邻居
        if u == end_idx:
            continue
            
        x, y = index_to_coord(u)
        
        for dx, dy in directions:
            # 如果开启了防止反复横跳功能，且当前方向与上一步方向相反，则跳过
            if prevent_back_and_forth and prev_direction[u] is not None:
                if (dx, dy) == (-prev_direction[u][0], -prev_direction[u][1]):
                    continue
                    
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                v = coord_to_index(nx, ny)
                # 确保不会从终点出发继续移动
                if v == end_idx and u != end_idx:
                    if dist[v] > dist[u] + matrix[nx][ny]:
                        dist[v] = dist[u] + matrix[nx][ny]
                        prev[v] = u
                        prev_direction[v] = (dx, dy)
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
                            count[v] += 1
                            # 检测负环
                            if count[v] > size:
                                # 找出负环路径
                                cycle = []
                                current = v
                                for _ in range(size):
                                    cycle.append(index_to_coord(current))
                                    current = prev[current]
                                    if current == v:
                                        break
                                cycle.append(index_to_coord(current))
                                cycle.reverse()
                                warning_msg = f"检测到负环？: {cycle}"
                                if gui_mode:
                                    buf = BytesIO()
                                    visualize_negative_cycle(np.array(matrix), cycle, orrM)
                                    plt.savefig(buf, format='png')
                                    plt.close()
                                    warning_msg = f"检测到负环: {cycle}"
                                    return (None, [], buf, warning_msg)
                                else:
                                    visualize_negative_cycle(np.array(matrix), cycle, orrM)
                                    warning_msg = f"检测到负环: {cycle}"
                                    print(f"警告: {warning_msg}")
                                    return (None, [])
                elif v != end_idx:
                    if dist[v] > dist[u] + matrix[nx][ny]:
                        dist[v] = dist[u] + matrix[nx][ny]
                        prev[v] = u
                        prev_direction[v] = (dx, dy)
                        if not in_queue[v]:
                            queue.append(v)
                            in_queue[v] = True
                            count[v] += 1
                            # 检测负环
                            if count[v] > size:
                                # 找出负环路径
                                cycle = []
                                current = v
                                for _ in range(size):
                                    cycle.append(index_to_coord(current))
                                    current = prev[current]
                                    if current == v:
                                        break
                                cycle.append(index_to_coord(current))
                                cycle.reverse()
                                # 可视化负环
                                if gui_mode:
                                    buf = BytesIO()
                                    visualize_negative_cycle(np.array(matrix), cycle, orrM)
                                    plt.savefig(buf, format='png')
                                    plt.close()
                                    warning_msg = f"检测到负环: {cycle}"
                                    return (None, [], buf, warning_msg)
                                else:
                                    visualize_negative_cycle(np.array(matrix), cycle, orrM)
                                    warning_msg = f"检测到负环: {cycle}"
                                    print(f"警告: {warning_msg}")
                                    return (None, [])
    
    # 重建路径
    end_idx = coord_to_index(end[0], end[1])
    if dist[end_idx] == float('inf'):
        return (None, [], None, "未找到路径") if gui_mode else (None, [])
    
    path = []
    u = end_idx
    while u != -1:
        path.append(index_to_coord(u))
        u = prev[u]
    path.reverse()
    
    # 处理GUI模式
    if gui_mode:
        buf = BytesIO()
        visualize_spfa(np.array(matrix), path, orrM)
        plt.savefig(buf, format='png')
        plt.close()
        return (dist[end_idx], path, buf, None)
    
    visualize_spfa(np.array(matrix), path, orrM)
    return (dist[end_idx], path)

# 示例用法
if __name__ == "__main__":
    # 示例矩阵
    orrM = np.array([
        [0, 5, 3, 0],
        [4, -2, 1, -3],
        [0, -4, 6, 5],
        [3, -2, -1, 20]
    ])
    M = orrM - 6
    matrix = -M

    start = (0, 0)
    end = (3, 2)
    
    # 传入原始矩阵用于显示
    distance, path = spfa_shortest_path(matrix, start, end, orrM=orrM,prevent_back_and_forth=False)
    print(f"最短距离: {distance}")
    print(f"最高得分: -----{distance}-----")
    print(f"路径: {path}")