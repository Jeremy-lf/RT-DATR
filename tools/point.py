from matplotlib.font_manager import font_scalings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#下面两句代码防止中文显示成方块
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# """
plt.rcParams['axes.titlesize'] = 8  # 标题字体大小
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8  # 图例字体大小
plt.rcParams['figure.dpi'] = 1000 #图片像素
# plt.rcParams['figure.figsize'] = (3.71, 3.71)
#2021
#SFA  41.3  UMT 41.7 

# 2022
# TIA 42.3,  OADA 45.4, MTTrans 43.4  O2Net 46.8, AQT 47.1

# 2023
# CMT 50.3  HT 50.4  MRT 51.2

# 2024 CAT 52.5
plt.scatter(2021.8, 41.3, color='orange', marker='o', s=14)  # done
plt.text(2021.8-0.05, 41.3-0.5, 'SFA', color='black', fontsize=6)

plt.scatter(2022.6, 41.7, color='darkcyan', marker='o', s=14) # done
plt.text(2022.6-0.05, 41.7-0.5, 'UMT', color='black', fontsize=6)

plt.scatter(2022.3, 42.3, color='darkcyan', marker='o', s=14) # done
plt.text(2022.3-0.05, 42.3-0.5, 'TIA', color='black', fontsize=6)

plt.scatter(2022.7, 45.4, color='darkcyan', marker='o', s=14) # done
plt.text(2022.7-0.1, 45.4-0.5, 'OADA', color='black', fontsize=6)

plt.scatter(2022.5, 43.4, color='orange', marker='o', s=14)
plt.text(2022.5-0.1, 43.4-0.5, 'MTTrans', color='black', fontsize=6)

plt.scatter(2022.4, 46.8, color='orange', marker='o', s=14)
plt.text(2022.4-0.1, 46.8-0.5, 'O2Net', color='black', fontsize=6)

plt.scatter(2022.6, 47.1, color='orange', marker='o', s=14)
plt.text(2022.6-0.05, 47.1-0.5, 'AQT', color='black', fontsize=6)

plt.scatter(2023.5, 50.3, color='darkcyan', marker='o', s=14)
plt.text(2023.5-0.05, 50.3-0.5, 'CMT', color='black', fontsize=6)

plt.scatter(2023.8, 50.4, color='darkcyan', marker='o', s=14)
plt.text(2023.8, 50.4-0.5, 'HT', color='black', fontsize=6)

plt.scatter(2023.9, 40.8, color='red', marker='*', s=14)
plt.text(2023.9-0.1, 40.8-0.5, 'ConfMix', color='black', fontsize=6)

plt.scatter(2024.45, 52.5, color='darkcyan', marker='o', s=14)
plt.text(2024.45-0.05, 52.5-0.5, 'CAT', color='black', fontsize=6)

plt.scatter(2024.7, 51.2, color='orange', marker='o', s=14)
plt.text(2024.7-0.05, 51.2-0.5, 'MRT', color='black', fontsize=6)


plt.scatter(2024.75, 52.9, color='red', marker='*')
plt.text(2024.535, 52.7-0.4, 'RT-DATR(Ours)', color='black', fontsize=6)


# 添加图例说明
x1 = [2023.8]
y1 = [50.4]
plt.scatter(x1, y1, label='CNN-based Detector', color='darkcyan', marker='o', s=6)

x2 = [2022.6]
y2 = [47.1]
plt.scatter(x2, y2, label='Transformer-based Detector', color='orange', marker='o', s=6)

x = [2024.75]
y = [52.9]
plt.scatter(x, y, label='Real-time Detector', color='red', marker='*', s=6)

x_bits = [2022, 2023, 2024,2025]
plt.xticks(x_bits)  # 横轴只有这四个刻度
# plt.yticks(y_bits)
# plt.ylim(42.0, 56.0)      #y坐标范围

# 画折线图
import numpy as np
from scipy.interpolate import make_interp_spline

xx = np.array([2021.8, 2022.4, 2022.6, 2023.5, 2024.45, 2024.75])
yy = np.array([41.3, 46.8, 47.1, 50.3, 52.5, 52.9])
m = make_interp_spline(xx, yy)

xnew = np.linspace(min(xx), 2024.8, 300) #300 represents number of points to make between T.min and T.max
power_smooth = m(xnew)
plt.plot(xnew, power_smooth, color='green', linestyle='-', linewidth=1)


plt.xlabel('Years')
plt.ylabel('mAP')
plt.tick_params(axis='both', which='major', labelsize=8)

plt.legend(fontsize=8)

plt.savefig('rtdatr-line.pdf', bbox_inches='tight')


# darkblue, darkcyan, darkblue, lightskyblue
# import matplotlib.pyplot as plt
 
# # 示例数据
# x = [1, 2, 3, 4, 5]
# y = [2, 3, 5, 7, 11]
# labels = ['A', 'B', 'C', 'D', 'E']
 
# # 创建散点图
# plt.scatter(x, y, color='blue', marker='o')
 
# # 在每个点旁边添加文本说明
# for i, label in enumerate(labels):
#     plt.text(x[i], y[i], label, fontsize=9, ha='right', va='bottom')
 
# # 设置标题和标签
# plt.title('Scatter Plot with Labels')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.savefig('rtdatr-line1.png')


# # x_bits = [12, 24, 32, 48]
# def joint_early():
#     with PdfPages('latency.pdf') as pdf:
#         # 开始绘制
#         colors = ['red', 'orange', 'blue', 'green', 'black', 'purple','brown']
#         markers = ['o', 'v', '*', 's', '+', 'd', '^']

#         x_2 = [ 3.42, 5.63, 9.02]
#         y_2 = [44.3, 49.1, 51.8]
#         plt.plot(x_2, y_2, color=colors[2], 
#                      marker=markers[2], linestyle='--', label='YOLOV6-3.0', markersize=3)

#         x_1 = [7.07, 9.5, 12.39, 16.86]
#         y_1 = [44.9, 50.6, 52.9, 53.9]
#         plt.plot(x_1, y_1, color=colors[1], 
#                      marker=markers[1], linestyle='--', label='YOLOv8', markersize=3)

        
#         x_1 = [10.57]#, 12.5, 15.0, 17.5, 20.0]
#         y_1 = [52.5]
#         plt.plot(x_1, y_1, color=colors[4], 
#                      marker=markers[4], linestyle='--', label='YOLOv9', markersize=3)


#         x_7 = [3.82, 6.39, 10.65]
#         y_7 = [45.4, 49.8, 51.8]
#         plt.plot(x_7, y_7, color=colors[6], 
#                      marker=markers[6], linestyle='--', label='Gold-YOLO', markersize=3)


#         x_6 = [4.8, 7.85, 10.6, 17.05]
#         y_6 = [43.1, 48.9, 51.4, 52.2]
#         plt.plot(x_6, y_6, color=colors[5], 
#                      marker=markers[5], linestyle='--', label='PP-YOLOE', markersize=3)

        
#         x_0 = [4.58,  6.90, 9.20, 13.71]#, 12.5, 15.0, 17.5, 20.0]
#         y_0 = [46.5,  51.3, 53.1, 54.3]
#         plt.plot(x_0, y_0, color=colors[3], 
#                      marker=markers[0], linestyle='--', label='RT-DETR', markersize=3)

        
#         x_3 = [4.58, 6.89, 9.20, 13.71]#, 12.5, 15.0, 17.5, 20.0]
#         y_3 = [48.7, 51.8, 53.5, 54.6]
#         plt.plot(x_3, y_3, color=colors[0], 
#                      marker=markers[3], linestyle='-', label='RT-DETRv3(ours)', markersize=3)
        
#         texts = ['R18', 'R50', 'R50','R101']
#         dxy = [(1.05, 0.4), (1.5, 0.55), (0.65, 0.4), (0.65, 0.4) ]
#         plt.legend(fontsize=8)
#         # 一共有多少个点就循环多少次
#         for i in range(len(x_3)):
#             plt.text(x_3[i] - dxy[i][0], y_3[i] + dxy[i][1], texts[i], color=colors[0], fontsize=6)


#         x_bits = [4, 8,  12,  16, 20]
#         plt.xticks(x_bits)  # 横轴只有这四个刻度
#         # plt.yticks(y_bits)
#         plt.ylim(42.0, 56.0)      #y坐标范围
#         plt.title("MS COCO Object Detection", fontsize=10)
#         plt.xlabel("Latency T4 TensorRT FP16(ms)")  # 作用为横坐标轴添加标签  fontsize=12
#         plt.ylabel("COCO AP(%)")  # 作用为纵坐标轴添加标签
#         plt.legend(fontsize=8)
#         plt.text(6.89-1.85, 51.8, 'Scaled', color=colors[0],fontsize=6)
#         # plt.show()
#         pdf.savefig(bbox_inches='tight')  # 如果要保存，就需要去掉plt.show()，因为plt.show()会把figure清除掉

# joint_early()




