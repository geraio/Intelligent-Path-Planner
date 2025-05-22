from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QComboBox, QPushButton, 
                            QGroupBox, QTabWidget, QTextEdit, QProgressBar,
                            QSplitter, QScrollArea, QCheckBox,QDoubleSpinBox)
from PyQt5.QtCore import Qt, QTimer 
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from io import BytesIO
from Qlearn import rl, draw
from PyQt5.QtWidgets import QSpinBox
import numpy as np  # 添加这行

class AlgorithmController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能路径规划控制器")
        self.setGeometry(100, 100, 800, 600)
        
        # 主界面布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        
        # 创建分割器
        # 创建分割器并设置初始比例
        splitter = QSplitter(Qt.Vertical)
        splitter.setSizes([200, 400])  # 上部控制面板200像素，下部结果区域400像素
        
        # 上部区域 - 控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        control_layout.addWidget(self.tab_widget)
        
        # 添加Q-learning标签页
        self.setup_qlearning_tab()
        self.setup_bfs_tab()
        self.setup_spfa_tab()
        self.setup_floyd_tab()
        self.setup_placeholder_tabs()
        
        control_panel.setLayout(control_layout)
        
        # 下部区域 - 结果显示
        result_panel = QWidget()
        result_layout = QVBoxLayout()
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        result_layout.addWidget(self.progress_bar)
        
        # 训练过程输出
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        result_layout.addWidget(QLabel("训练过程输出:"))
        result_layout.addWidget(self.output_text)
        
        # 结果图片显示 - 修改这部分
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(800, 600)  # 增大最小尺寸
        scroll = QScrollArea()
        scroll.setWidget(self.result_label)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(500)  # 增大滚动区域最小高度
        result_layout.addWidget(QLabel("结果可视化:"))
        result_layout.addWidget(scroll)
        
        result_panel.setLayout(result_layout)
        
        # 添加部件到分割器
        splitter.addWidget(control_panel)
        splitter.addWidget(result_panel)
        
        self.main_layout.addWidget(splitter)
        
        # 重定向标准输出
        import sys
        from io import StringIO
        self.stdout = sys.stdout
        self.string_io = StringIO()
        sys.stdout = self.string_io
        
        # 定时器用于更新输出
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_output)
        self.timer.start(100)  # 每100ms更新一次
    
    def update_output(self):
        # 更新控制台输出
        value = self.string_io.getvalue()
        if value:
            self.output_text.append(value)
            self.string_io.truncate(0)
            self.string_io.seek(0)
    
    def closeEvent(self, event):
        # 恢复标准输出
        import sys
        sys.stdout = self.stdout
        super().closeEvent(event)
    
    def setup_qlearning_tab(self):
        """设置Q-learning算法的控制界面"""
        qlearning_tab = QWidget()
        layout = QVBoxLayout()
        
        # 参数设置组
        params_group = QGroupBox("Q-learning参数设置")
        params_layout = QVBoxLayout()
        
        # 算法参数选择
        self.add_parameter_control(params_layout, "算法类型:", ["SA", "QL", "EXP"], "SA")
        self.add_parameter_control(params_layout, "选择方式:", ["Q", "S", "R"], "Q")
        self.add_parameter_control(params_layout, "边界处理:", ["Not", "Try"], "Not")
        self.add_parameter_control(params_layout, "回溯处理:", ["Not", "Never", "Allow"], "Not")
        
        # 添加迭代次数输入
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("迭代次数:"))
        self.iterations_spinbox = QSpinBox()
        self.iterations_spinbox.setRange(1, 10000)
        self.iterations_spinbox.setValue(100)
        h_layout.addWidget(self.iterations_spinbox)
        params_layout.addLayout(h_layout)
        
        # 添加减数输入
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("奖励减数:"))
        self.subtract_spinbox = QDoubleSpinBox()
        self.subtract_spinbox.setDecimals(2)
        self.subtract_spinbox.setRange(0.0, 100.0)
        self.subtract_spinbox.setValue(6.0)
        h_layout.addWidget(self.subtract_spinbox)
        params_layout.addLayout(h_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 运行按钮
        run_button = QPushButton("运行Q-learning")
        run_button.clicked.connect(self.run_qlearning)
        layout.addWidget(run_button)
        
        qlearning_tab.setLayout(layout)
        self.tab_widget.addTab(qlearning_tab, "Q-learning")
    
    def setup_bfs_tab(self):
        """设置BFS算法的控制界面"""
        bfs_tab = QWidget()
        layout = QVBoxLayout()
        
        # 参数设置组
        params_group = QGroupBox("BFS参数设置")
        params_layout = QVBoxLayout()
        
        # 添加矩阵输入
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("矩阵(每行用逗号分隔):"))
        self.matrix_input = QTextEdit()
        self.matrix_input.setMaximumHeight(100)
        self.matrix_input.setPlaceholderText("例如:\n0,5,3,0\n4,-2,1,-3\n0,-4,6,5\n3,-2,-1,20")
        h_layout.addWidget(self.matrix_input)
        params_layout.addLayout(h_layout)
        
        # 添加起点和终点输入
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("起点(行,列):"))
        self.start_input = QTextEdit()
        self.start_input.setMaximumHeight(30)
        self.start_input.setPlaceholderText("例如: 0,0")
        h_layout.addWidget(self.start_input)
        
        h_layout.addWidget(QLabel("终点(行,列):"))
        self.end_input = QTextEdit()
        self.end_input.setMaximumHeight(30)
        self.end_input.setPlaceholderText("例如: 3,3")
        h_layout.addWidget(self.end_input)
        params_layout.addLayout(h_layout)
        
        # 添加惩罚系数输入
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("惩罚系数:"))
        self.penalty_spinbox = QSpinBox()
        self.penalty_spinbox.setRange(0, 100)
        self.penalty_spinbox.setValue(6)  # 默认值为1
        h_layout.addWidget(self.penalty_spinbox)
        params_layout.addLayout(h_layout)
        
        # 添加复选框参数
        self.allow_revisit_check = QCheckBox("允许重复访问")
        params_layout.addWidget(self.allow_revisit_check)
        
        self.prevent_back_forth_check = QCheckBox("防止反复横跳")
        params_layout.addWidget(self.prevent_back_forth_check)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 运行按钮
        run_button = QPushButton("运行BFS算法")
        run_button.clicked.connect(self.run_bfs)
        layout.addWidget(run_button)
        
        bfs_tab.setLayout(layout)
        self.tab_widget.addTab(bfs_tab, "BFS")
    
    def setup_spfa_tab(self):
        """设置SPFA算法的控制界面"""
        spfa_tab = QWidget()
        layout = QVBoxLayout()
        
        # 参数设置组
        params_group = QGroupBox("SPFA参数设置")
        params_layout = QVBoxLayout()
        
        # 添加矩阵输入 - 使用spfa前缀
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("矩阵(每行用逗号分隔):"))
        self.spfa_matrix_input = QTextEdit()
        self.spfa_matrix_input.setMaximumHeight(100)
        self.spfa_matrix_input.setPlaceholderText("例如:\n0,5,3,0\n4,-2,1,-3\n0,-4,6,5\n3,-2,-1,20")
        h_layout.addWidget(self.spfa_matrix_input)
        params_layout.addLayout(h_layout)
        
        # 添加起点和终点输入 - 使用spfa前缀
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("起点(行,列):"))
        self.spfa_start_input = QTextEdit()
        self.spfa_start_input.setMaximumHeight(30)
        self.spfa_start_input.setPlaceholderText("例如: 0,0")
        h_layout.addWidget(self.spfa_start_input)
        
        h_layout.addWidget(QLabel("终点(行,列):"))
        self.spfa_end_input = QTextEdit()
        self.spfa_end_input.setMaximumHeight(30)
        self.spfa_end_input.setPlaceholderText("例如: 3,3")
        h_layout.addWidget(self.spfa_end_input)
        params_layout.addLayout(h_layout)
        
        # 添加复选框参数 - 使用spfa前缀
        self.spfa_prevent_back_forth_check = QCheckBox("防止反复横跳")
        params_layout.addWidget(self.spfa_prevent_back_forth_check)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # 运行按钮
        run_button = QPushButton("运行SPFA算法")
        run_button.clicked.connect(self.run_spfa)
        layout.addWidget(run_button)
        
        spfa_tab.setLayout(layout)
        self.tab_widget.addTab(spfa_tab, "SPFA")
    
    def setup_floyd_tab(self):
        """设置Floyd算法的控制界面"""
        floyd_tab = QWidget()
        layout = QVBoxLayout()
        
        # 参数设置组
        params_group = QGroupBox("Floyd参数设置")
        params_layout = QVBoxLayout()
        
        # 添加矩阵输入
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("矩阵(每行用逗号分隔):"))
        self.floyd_matrix_input = QTextEdit()
        self.floyd_matrix_input.setMaximumHeight(100)
        self.floyd_matrix_input.setPlaceholderText("例如:\n0,5,3,0\n4,-2,1,-3\n0,-4,6,5\n3,-2,-1,20")
        h_layout.addWidget(self.floyd_matrix_input)
        params_layout.addLayout(h_layout)
        
        # 添加起点和终点输入
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("起点(行,列):"))
        self.floyd_start_input = QTextEdit()
        self.floyd_start_input.setMaximumHeight(30)
        self.floyd_start_input.setPlaceholderText("例如: 0,0")
        h_layout.addWidget(self.floyd_start_input)
        
        h_layout.addWidget(QLabel("终点(行,列):"))
        self.floyd_end_input = QTextEdit()
        self.floyd_end_input.setMaximumHeight(30)
        self.floyd_end_input.setPlaceholderText("例如: 3,3")
        h_layout.addWidget(self.floyd_end_input)
        params_layout.addLayout(h_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # 运行按钮
        run_button = QPushButton("运行Floyd算法")
        run_button.clicked.connect(self.run_floyd)
        layout.addWidget(run_button)
        
        floyd_tab.setLayout(layout)
        self.tab_widget.addTab(floyd_tab, "Floyd")

    def setup_placeholder_tabs(self):
        """为其他算法添加占位标签页"""
        
        # 示例: 添加MCTS算法的占位标签页
        mcts_tab = QWidget()
        mcts_layout = QVBoxLayout()
        mcts_layout.addWidget(QLabel("蒙特卡洛树搜索(MCTS)参数设置\n(待实现)"))
        mcts_tab.setLayout(mcts_layout)
        self.tab_widget.addTab(mcts_tab, "MCTS")
    
    def add_parameter_control(self, layout, label_text, options, default):
        """添加参数控制组件"""
        h_layout = QHBoxLayout()
        label = QLabel(label_text)
        combo = QComboBox()
        combo.addItems(options)
        combo.setCurrentText(default)
        
        h_layout.addWidget(label)
        h_layout.addWidget(combo)
        layout.addLayout(h_layout)
        
        # 保存控件引用
        setattr(self, f"ql_{label_text.replace(':', '').lower()}_combo", combo)
    
    def run_qlearning(self):
        """运行Q-learning算法"""
        options = {
            'func': self.ql_算法类型_combo.currentText(),
            'way': self.ql_选择方式_combo.currentText(),
            'wall': self.ql_边界处理_combo.currentText(),
            'back': self.ql_回溯处理_combo.currentText(),
            'Video': 'Not',
            'iterations': self.iterations_spinbox.value(),
            'MAX_EPISODES': self.iterations_spinbox.value(),
            'subtract': self.subtract_spinbox.value(),
            'warning_callback': self.handle_warning  # 添加警告回调
        }
        
        # 清空之前的输出
        self.output_text.clear()
        self.progress_bar.setValue(0)
        
        # 定义进度回调函数
        def update_progress(progress):
            self.progress_bar.setValue(progress)
            QApplication.processEvents()  # 更新UI
            
        # 运行算法并传入进度回调
        q_table = rl(options, progress_callback=update_progress)
        
        print(q_table)
        # 绘制结果并显示
        self.draw_result(q_table, options)
        
        # 更新进度条
        self.progress_bar.setValue(100)
    
    def handle_warning(self, warning_msg):
        """处理警告信息"""
        self.output_text.append(f"警告: {warning_msg}")
        QApplication.processEvents()  # 更新UI
    
    def draw_result(self, q_table, options):
        """绘制结果并显示在GUI中"""
        # 保存图片到内存
        buf = BytesIO()
        plt.figure(figsize=(10, 8))
        
        # 调用原来的draw函数
        draw(q_table, options)
        
        # 保存到缓冲区
        plt.savefig(buf, format='png')
        plt.close()
        
        # 显示图片
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        self.result_label.setPixmap(pixmap)
        buf.close()

    def run_bfs(self):
        """运行BFS算法"""
        try:
            # 默认矩阵
            orrM = np.array([
                [0, 5, 3, 0],
                [4, -2, 1, -3],
                [0, -4, 6, 5],
                [3, -2, -1, 20]
            ])
            # default_matrix = orrM - 6  # 修改这里，直接使用原始矩阵减去6
            # default_matrix = -default_matrix  # 取负值
            default_matrix = orrM
            
            # 默认起点和终点
            default_start = (0, 0)
            default_end = (3, 3)
            
            # 解析矩阵输入
            matrix_text = self.matrix_input.toPlainText().strip()
            matrix = []
            
            if matrix_text:  # 如果有输入则解析输入
                for line in matrix_text.split('\n'):
                    # 跳过空行
                    if not line.strip():
                        continue
                    # 过滤空值
                    row = [float(x.strip()) for x in line.split(',') if x.strip()]
                    if row:  # 只添加非空行
                        matrix.append(row)
            else:  # 没有输入则使用默认值s
                print('使用默认值')
                matrix = default_matrix
                self.matrix_input.setPlainText("\n".join([",".join(map(str, row)) for row in default_matrix]))
            
            # 修改矩阵是否为空的判断
            if isinstance(matrix, np.ndarray):
                if matrix.size == 0:
                    raise ValueError("矩阵不能为空")
            else:
                if not matrix:
                    raise ValueError("矩阵不能为空")
            
            # 解析起点和终点
            start_text = self.start_input.toPlainText().strip()
            end_text = self.end_input.toPlainText().strip()
            
            # 处理起点
            if start_text:
                start = tuple(int(x.strip()) for x in start_text.split(',') if x.strip())
                if len(start) != 2:
                    raise ValueError("起点必须是两个数字，用逗号分隔")
            else:
                start = default_start
                self.start_input.setPlainText(f"{default_start[0]},{default_start[1]}")
            
            # 处理终点
            if end_text:
                end = tuple(int(x.strip()) for x in end_text.split(',') if x.strip())
                if len(end) != 2:
                    raise ValueError("终点必须是两个数字，用逗号分隔")
            else:
                end = default_end
                self.end_input.setPlainText(f"{default_end[0]},{default_end[1]}")
            
            if len(start) != 2 or len(end) != 2:
                raise ValueError("起点和终点必须是两个数字，用逗号分隔")
            
            # 获取复选框状态
            allow_revisit = self.allow_revisit_check.isChecked()
            prevent_back_forth = self.prevent_back_forth_check.isChecked()
            
            # 获取惩罚系数
            penalty = self.penalty_spinbox.value()
            
            # 运行BFS算法，添加penalty参数
            from BFS import shortest_path_with_visualization
            np_matrix = np.array(matrix, dtype=np.float64)
            distance, path, image_buf = shortest_path_with_visualization(
                np_matrix, start, end,
                allow_revisit=allow_revisit,
                prevent_back_and_forth=prevent_back_forth,
                orrM=orrM,
                gui_mode=True,
                penalty=penalty  # 添加惩罚系数参数
            )
            
            # 显示结果
            # self.output_text.append(f"最短距离: {distance}")
            self.output_text.append(f"最高得分: -----{distance}-----")
            self.output_text.append(f"路径: {path}")
            
            # 在GUI中显示图像
            if image_buf:
                pixmap = QPixmap()
                pixmap.loadFromData(image_buf.getvalue())
                self.result_label.setPixmap(pixmap)
                image_buf.close()
            
        except Exception as e:
            self.output_text.append(f"BFS错误: {str(e)}")

    def run_spfa(self):
        """运行SPFA算法"""
        try:
        # if True:
            # 默认矩阵
            orrM = np.array([
                [0, 5, 3, 0],
                [4, -2, 1, -3],
                [0, -4, 6, 5],
                [3, -2, -1, 20]
            ])
            default_matrix = orrM - 6
            default_matrix = -default_matrix
            
            # 默认起点和终点
            default_start = (0, 0)
            default_end = (3, 3)
            
            # 解析矩阵输入 - 使用spfa前缀
            matrix_text = self.spfa_matrix_input.toPlainText().strip()
            matrix = []
            
            if matrix_text:
                for line in matrix_text.split('\n'):
                    if not line.strip():
                        continue
                    row = [float(x.strip()) for x in line.split(',') if x.strip()]
                    if row:
                        matrix.append(row)
            else:
                matrix = default_matrix
                self.spfa_matrix_input.setPlainText("\n".join([",".join(map(str, row)) for row in default_matrix]))
            
            # 解析起点和终点 - 使用spfa前缀
            start_text = self.spfa_start_input.toPlainText().strip()
            end_text = self.spfa_end_input.toPlainText().strip()
            
            start = tuple(int(x.strip()) for x in start_text.split(',')) if start_text else default_start
            end = tuple(int(x.strip()) for x in end_text.split(',')) if end_text else default_end
            
            if len(start) != 2 or len(end) != 2:
                raise ValueError("起点和终点必须是两个数字，用逗号分隔")
            
            # 获取复选框状态 - 使用spfa前缀
            prevent_back_forth = self.spfa_prevent_back_forth_check.isChecked()
            
            # 运行SPFA算法
            from SPFA import spfa_shortest_path
            np_matrix = np.array(matrix, dtype=np.float64)
            # 修改调用方式，添加一个变量接收警告信息
            distance, path, image_buf, warning_msg = spfa_shortest_path(
            np_matrix, start, end,
            orrM=orrM,
            prevent_back_and_forth=prevent_back_forth,
            gui_mode=True
            )
            
            # 如果有警告信息则显示
            if warning_msg:
                self.output_text.append(f"警告: {warning_msg}")
            
            # 显示结果
            # self.output_text.append(f"最短距离: {distance}")
            self.output_text.append(f"最高得分: -----{distance}-----")
            self.output_text.append(f"路径: {path}")
            
            # 显示图像
            if image_buf:
                pixmap = QPixmap()
                pixmap.loadFromData(image_buf.getvalue())
                self.result_label.setPixmap(pixmap)
                image_buf.close()
                
        except Exception as e:
            self.output_text.append(f"SPFA错误: {str(e)}")


    def run_floyd(self):
        """运行Floyd算法"""
        # try:
        # 默认矩阵
        orrM = np.array([
            [0, 5, 3, 0],
            [4, -2, 1, -3],
            [0, -4, 6, 5],
            [3, -2, -1, 20]
        ])
        default_matrix = orrM - 6
        default_matrix = -default_matrix
        
        # 默认起点和终点
        default_start = (0, 0)
        default_end = (3, 3)
        
        # 解析矩阵输入
        matrix_text = self.floyd_matrix_input.toPlainText().strip()
        matrix = []
        
        if matrix_text:
            for line in matrix_text.split('\n'):
                if not line.strip():
                    continue
                row = [float(x.strip()) for x in line.split(',') if x.strip()]
                if row:
                    matrix.append(row)
        else:
            matrix = default_matrix
            self.floyd_matrix_input.setPlainText("\n".join([",".join(map(str, row)) for row in default_matrix]))
        
        if isinstance(matrix, np.ndarray):
            if matrix.size == 0:
                raise ValueError("矩阵不能为空")
        else:
            if not matrix:
                raise ValueError("矩阵不能为空")
        
        # 解析起点和终点
        start_text = self.floyd_start_input.toPlainText().strip()
        end_text = self.floyd_end_input.toPlainText().strip()
        
        # 处理起点
        if start_text:
            start = tuple(int(x.strip()) for x in start_text.split(',') if x.strip())
            if len(start) != 2:
                raise ValueError("起点必须是两个数字，用逗号分隔")
        else:
            start = default_start
            self.floyd_start_input.setPlainText(f"{default_start[0]},{default_start[1]}")
        

        # 处理终点
        if end_text:
            end = tuple(int(x.strip()) for x in end_text.split(',') if x.strip())
            if len(end) != 2:
                raise ValueError("终点必须是两个数字，用逗号分隔")
        else:
            end = default_end
            self.floyd_end_input.setPlainText(f"{default_end[0]},{default_end[1]}")
        
        if len(start) != 2 or len(end) != 2:
            raise ValueError("起点和终点必须是两个数字，用逗号分隔")

        # 运行Floyd算法，添加gui_mode=True参数
        from Floyd import floyd_shortest_path
        np_matrix = np.array(matrix, dtype=np.float64)
        distance, path, image_buf = floyd_shortest_path(
            np_matrix, start, end, 
            orrM=orrM,
            gui_mode=True,  # 添加这个参数
        )
        
        # 显示结果
        # self.output_text.append(f"最短距离: {distance}")
        self.output_text.append(f"最高得分: -----{distance}-----")
        self.output_text.append(f"路径: {path}")
        
        # 在GUI中显示图像
        if image_buf:
            pixmap = QPixmap()
            pixmap.loadFromData(image_buf.getvalue())
            self.result_label.setPixmap(pixmap)
            image_buf.close()
            
        # except Exception as e:
        #     self.output_text.append(f"Floyd错误: {str(e)}")

if __name__ == "__main__":
    app = QApplication([])
    window = AlgorithmController()
    window.show()
    app.exec_()