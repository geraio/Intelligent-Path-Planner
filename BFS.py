from io import BytesIO  # 添加这行
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

def visualize_search(matrix, path_history, best_path, orrM=None, gui_mode=False):
    """
    可视化搜索过程和最终结果
    
    参数:
        matrix: 转换后的矩阵
        path_history: 搜索过程中访问的所有路径历史
        best_path: 最终找到的最短路径
        orrM: 原始矩阵(可选)
        gui_mode: 是否在GUI模式下运行(不显示动画窗口)
    """
    if not gui_mode:
        # 原有逻辑保持不变
        fig, ax = plt.subplots(figsize=(8, 6))  # 从(10,8)改为(8,6)
        
        # 创建颜色映射，突出显示不同值
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
        
        # 在网格中显示数值
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i][j]:.1f}", 
                       ha="center", va="center", color="w", fontsize=12)
        
        # 标记起点和终点
        start = path_history[0][0]
        end = path_history[0][-1]
        ax.add_patch(patches.Rectangle((start[1]-0.5, start[0]-0.5), 1, 1, 
                                      linewidth=2, edgecolor='r', facecolor='none'))
        ax.add_patch(patches.Rectangle((end[1]-0.5, end[0]-0.5), 1, 1, 
                                      linewidth=2, edgecolor='g', facecolor='none'))
        
        # 动画函数
        def update(frame):
            ax.set_title(f"Step {frame+1}/{len(path_history)}")
            path = path_history[frame]
            
            # 清除之前的路径
            for patch in ax.patches[2:]:
                patch.remove()
            
            # 绘制当前路径
            for i in range(len(path)-1):
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                ax.arrow(y1, x1, y2-y1, x2-x1, 
                        head_width=0.2, head_length=0.2, 
                        fc='yellow', ec='yellow', alpha=0.5)
            
            return ax.patches
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=len(path_history), 
                            interval=200, repeat=False)
        
        # 显示最终最优路径
        plt.figure(figsize=(8, 6))
        im = plt.imshow(matrix, cmap=cmap, norm=norm)
        plt.colorbar(im, label='Value')
        plt.grid(which="minor", color="w", linestyle='-', linewidth=2)
        
        # 显示数值 - 使用orrM如果提供了的话
        display_matrix = orrM if orrM is not None else matrix
        for i in range(display_matrix.shape[0]):
            for j in range(display_matrix.shape[1]):
                plt.text(j, i, f"{display_matrix[i][j]:.1f}", 
                        ha="center", va="center", color="w", fontsize=12)
        
        # 绘制最优路径
        for i in range(len(best_path)-1):
            x1, y1 = best_path[i]
            x2, y2 = best_path[i+1]
            plt.arrow(y1, x1, y2-y1, x2-x1, 
                     head_width=0.3, head_length=0.3, 
                     fc='red', ec='red', linewidth=2)
        
        plt.title("Final Shortest Path")
        plt.show()
        return None
    else:
        # GUI模式下的逻辑 - 只生成最终结果图
        fig, ax = plt.subplots(figsize=(8, 6))  # 从(10,8)改为(8,6)
        
        # 创建颜色映射
        cmap = plt.cm.viridis
        norm = colors.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
        
        # 绘制矩阵和colorbar
        im = ax.imshow(matrix, cmap=cmap, norm=norm)
        
        # 添加网格线
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        
        # 显示数值 - 使用orrM如果提供了的话
        display_matrix = orrM if orrM is not None else matrix
        for i in range(display_matrix.shape[0]):
            for j in range(display_matrix.shape[1]):
                ax.text(j, i, f"{display_matrix[i][j]:.1f}", 
                       ha="center", va="center", color="w", fontsize=12)
        
        # 绘制最优路径
        for i in range(len(best_path)-1):
            x1, y1 = best_path[i]
            x2, y2 = best_path[i+1]
            ax.arrow(y1, x1, y2-y1, x2-x1, 
                    head_width=0.3, head_length=0.3, 
                    fc='red', ec='red', linewidth=2)
        
        # 将图像保存到缓冲区并返回
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

# 修改后的shortest_path函数，添加路径记录功能
def shortest_path_with_visualization(matrix, start, end, allow_revisit=False, 
                                   max_steps=500, orrM=None, 
                                   prevent_back_and_forth=False, gui_mode=False,penalty=6):
    matrix = matrix-penalty
    matrix = -matrix
    rows = len(matrix)
    if rows == 0:
        return (float('inf'), [])
    cols = len(matrix[0])
    
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    min_distance = float('inf')
    best_path = []
    path_history = []  # 记录所有访问路径
    
    # 计算矩阵中的最大值作为阈值基准
    max_matrix_value = max(max(row) for row in matrix)
    threshold = max_matrix_value * 3
    printed_prune_message = False  # 添加这个变量
    
    def bfs(x, y, current_distance, path, steps, last_direction=None):
        nonlocal min_distance, best_path, path_history
        nonlocal printed_prune_message  # 添加这个nonlocal声明
        
        # 如果超出边界或超过最大步数，返回
        if x < 0 or x >= rows or y < 0 or y >= cols or steps > max_steps:
            return
            
        # 如果不允许重复访问且已访问过，返回
        if not allow_revisit and visited[x][y]:
            return
            
        # 剪枝条件：当前距离已经超过当前最短距离+阈值
        if min_distance != float('inf') and current_distance > min_distance + threshold:
            if not printed_prune_message:
                print("已超过最大允许深度，程序自动剪枝")
                printed_prune_message = True
            return
            
        # 到达终点，检查是否为更短路径
        if (x, y) == end:
            if current_distance < min_distance:
                min_distance = current_distance
                best_path = path.copy()
            return
            
        # 标记为已访问
        visited[x][y] = True
        path.append((x, y))
        
        # 向四个方向搜索，传递steps+1
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            # 如果开启了防止反复横跳功能，且当前方向与上一步相反，则跳过
            if prevent_back_and_forth and last_direction is not None:
                if (dx, dy) == (-last_direction[0], -last_direction[1]):
                    continue
                    
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if allow_revisit or not visited[nx][ny]:
                    bfs(nx, ny, current_distance + matrix[nx][ny], path, steps + 1, (dx, dy))
        
        # 回溯
        visited[x][y] = False
        path.pop()
        # 在每次路径变化时记录
        path_history.append(path.copy() + [(x, y)])
        
    # 调用原有逻辑，添加last_direction参数
    start_x, start_y = start
    bfs(start_x, start_y, 0, [], 0, None)
    
    if min_distance == float('inf'):
        return (None, [], None) if gui_mode else (None, [])
    
    # 添加终点到最佳路径
    final_path = best_path + [end]
    path_history.append(final_path)
    
    # 可视化
    if gui_mode:
        image_buf = visualize_search(np.array(matrix), path_history, final_path, orrM, gui_mode=True)
        return (min_distance, final_path, image_buf)
    else:
        visualize_search(np.array(matrix), path_history, final_path, orrM)
        return (min_distance, final_path)

# 示例用法
if __name__ == "__main__":
    # 示例矩阵
    orrM = np.array([
    [0, 5, 3, 0],
    [4, -2, 1, -3],
    [0, -4, 6, 5],
    [3, -2, -1, 20]
    ])
    # M = orrM-6
    # matrix = -M

    print(orrM)

    start = (0, 0)
    end = (3, 3)
    
    # 使用带可视化的版本，传入orrM
    distance, path = shortest_path_with_visualization(orrM, start, end, allow_revisit=True, orrM=orrM, prevent_back_and_forth=False)
    print(f"最短距离: {distance}")
    print(f"最高得分: -----{distance}-----")
    print(f"路径: {path}")