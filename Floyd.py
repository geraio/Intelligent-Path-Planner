from io import BytesIO  # 添加这行
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation

def visualize_floyd(matrix, path, orrM=None, gui_mode=False):
    """
    可视化Floyd算法找到的最短路径
    
    参数:
        matrix: 转换后的矩阵
        path: 找到的最短路径
        orrM: 原始矩阵(可选)
        gui_mode: 是否在GUI模式下运行(不显示窗口)
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
    
    plt.title("Floyd Algorithm - Shortest Path")
    if gui_mode:
        # GUI模式下返回图像数据
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf
    else:
        plt.show()
        return None

def floyd_shortest_path(matrix, start, end, orrM=None, gui_mode=False):
    """
    使用Floyd算法寻找矩阵中从起点到终点的最短路径
    
    参数:
        matrix: 二维矩阵，每个元素代表移动到此位置的距离
        start: 起点坐标 (x, y)
        end: 终点坐标 (x, y)
        orrM: 原始矩阵(可选)
        prevent_back_and_forth: 是否禁止反复横跳
    """
    rows = len(matrix)
    if rows == 0:
        return (float('inf'), [])
    cols = len(matrix[0])
    
    # 将二维坐标转换为一维索引
    def coord_to_index(x, y):
        return x * cols + y
    
    # 将一维索引转换为二维坐标
    def index_to_coord(index):
        return (index // cols, index % cols)
    
    size = rows * cols
    dist = np.full((size, size), float('inf'))
    next_node = np.full((size, size), -1, dtype=int)
    
    # 初始化距离矩阵
    for i in range(size):
        dist[i][i] = 0
    
    # 填充邻接矩阵
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for x in range(rows):
        for y in range(cols):
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    u = coord_to_index(x, y)
                    v = coord_to_index(nx, ny)
                    dist[u][v] = matrix[nx][ny]
                    next_node[u][v] = v
    
    # Floyd算法核心
    for k in range(size):
        for i in range(size):
            for j in range(size):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # 在Floyd算法核心部分后可添加负权环检测
    for i in range(size):
        if dist[i][i] < 0:  # 对角线出现负值表示有负权环
            print("警告：图中存在负权环，最短路径可能无界")
            break
    
    # 重建路径
    start_idx = coord_to_index(start[0], start[1])
    end_idx = coord_to_index(end[0], end[1])
    
    if next_node[start_idx][end_idx] == -1:
        return (float('inf'), [])  # 没有路径
    
    path = []
    u = start_idx
    count = 0
    while u != end_idx:
        x, y = index_to_coord(u)
        path.append((int(x), int(y)))
        u = next_node[u][end_idx]
        count += 1
        if count>1000:
            print("警告：路径过长，可能存在死循环")
            break
    x, y = index_to_coord(end_idx)
    path.append((int(x), int(y)))
    
    # 修正路径成本计算方式
    calculated_distance = 0  # 起点成本不计入
    for i in range(1, len(path)):  # 从第二个节点开始计算
        x, y = path[i]
        calculated_distance += matrix[x][y]
    
    # print(f"验证路径成本: {calculated_distance}")
    # 在返回前添加可视化调用
    if gui_mode:
        image_buf = visualize_floyd(np.array(matrix), path, orrM, gui_mode=True)
        return (calculated_distance, path, image_buf)
    else:
        visualize_floyd(np.array(matrix), path, orrM)
        return (calculated_distance, path)

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
    distance, path = floyd_shortest_path(matrix, start, end, orrM=orrM)
    print(f"最短距离: {distance}")
    print(f"最高得分: -----{distance}-----")
    print(f"路径: {path}")