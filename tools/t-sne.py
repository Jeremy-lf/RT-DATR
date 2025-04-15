from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import paddle
import os
from matplotlib.backends.backend_pdf import PdfPages

# files = os.listdir('./da_src_encoder_tsne/')
# S = []
# for file in files:
#     tmp = np.load('./da_src_encoder_tsne/' + file)
#     B, N, C = tmp.shape
#     tmp = tmp.reshape(B,-1)
#     S.append(tmp)


# files = os.listdir('./da_tgt_encoder_tsne/')
# T = []
# for file in files:
#     tmp = np.load('./da_tgt_encoder_tsne/' + file)
#     B, N, C = tmp.shape
#     tmp = tmp.reshape(B,-1)
#     T.append(tmp)


# S1 = np.concatenate(S, axis=0)
# np.save('da_src_encoder_tsne.npy', S1)
# T1 = np.concatenate(T, axis=0)
# np.save('da_tgt_encoder_tsne.npy', T1)



# with PdfPages('tsne-42-10.pdf') as pdf:
#     random_state = 42  # 可以是任何整数
#     S1 = np.load('src_encoder_tsne.npy')
#     T1 = np.load('tgt_encoder_tsne.npy')
#     S1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=10).fit_transform(S1)
#     T11_tsne = TSNE(n_components=2, random_state=random_state, perplexity=10).fit_transform(T1)

#     plt.scatter(S1_tsne[:, 0], S1_tsne[:, 1], s=8, label='source', color='red')
    
#     # 绘制data2的点
#     plt.scatter(T11_tsne[:, 0], T11_tsne[:, 1], s=8, label='target', color='blue')
    
#     # 添加图例
#     # plt.legend(loc='lower right')

#     # 不显示坐标
#     plt.xticks([])
#     plt.yticks([])
#     pdf.savefig(bbox_inches="tight")


# with PdfPages('tsne-42-25.pdf') as pdf:
#     random_state = 42  # 可以是任何整数
#     S1 = np.load('src_encoder_tsne.npy')
#     T1 = np.load('tgt_encoder_tsne.npy')
#     S1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=25).fit_transform(S1)
#     T11_tsne = TSNE(n_components=2, random_state=random_state, perplexity=25).fit_transform(T1)

#     plt.scatter(S1_tsne[:, 0], S1_tsne[:, 1], s=8, label='source', color='red')
    
#     # 绘制data2的点
#     plt.scatter(T11_tsne[:, 0], T11_tsne[:, 1], s=8, label='target', color='blue')
    
#     # 添加图例
#     # plt.legend(loc='lower right')

#     # 不显示坐标
#     plt.xticks([])
#     plt.yticks([])
#     pdf.savefig(bbox_inches="tight")


# ok 离散的
# with PdfPages('tsne-da-42-50.pdf') as pdf:
#     random_state = 42  # 可以是任何整数
#     S1 = np.load('da_src_encoder_tsne.npy')
#     T1 = np.load('da_tgt_encoder_tsne.npy')
#     S1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=50).fit_transform(S1)
#     T11_tsne = TSNE(n_components=2, random_state=random_state, perplexity=50).fit_transform(T1)

#     plt.scatter(S1_tsne[:, 0], S1_tsne[:, 1], s=8, label='source', color='red')
    
#     # 绘制data2的点
#     plt.scatter(T11_tsne[:, 0], T11_tsne[:, 1], s=8, label='target', color='blue')
    
#     # 添加图例
#     # plt.legend(loc='lower right')

#     # 不显示坐标
#     plt.xticks([])
#     plt.yticks([])
#     pdf.savefig(bbox_inches="tight")
    # plt.savefig('tsne-2.png',)
    # plt.show()





# import matplotlib.pyplot as plt

# # 创建一个包含两个子图的图形，1行2列布局
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# random_state = 42  # 可以是任何整数
# S1 = np.load('src_encoder_tsne.npy')
# T1 = np.load('tgt_encoder_tsne.npy')
# S1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=25).fit_transform(S1)
# T1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=25).fit_transform(T1)

# ax1.scatter(S1_tsne[:, 0], S1_tsne[:, 1], s=8, label='source', color='red')
# ax1.scatter(T1_tsne[:, 0], T1_tsne[:, 1], s=8, label='target', color='blue')
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_xlabel('RTDETR')
# ax1.set_ylabel('Encoder Feature')

# # 在第二个子图上绘制不同的数据
# DA_S1 = np.load('src_encoder_tsne.npy')
# DA_T1 = np.load('tgt_encoder_tsne.npy')
# DAS1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=10).fit_transform(DA_S1)
# DAT1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=10).fit_transform(DA_T1)

# ax2.scatter(DAS1_tsne[:, 0], DAS1_tsne[:, 1], s=8, label='source', color='red')
# ax2.scatter(DAT1_tsne[:, 0], DAT1_tsne[:, 1], s=8, label='target', color='blue')
# # 不显示坐标
# ax2.set_xticks([])
# ax2.set_yticks([])

# ax2.set_xlabel('RT-DATR(Ours)')

# # 添加图例
# # plt.legend(loc='lower right')

# # 调整子图之间的间距（可选）
# plt.tight_layout()

# # 将整个图形保存为PDF文件
# plt.savefig('feature_visualization_2.pdf', bbox_inches="tight")



# 创建一个包含两个子图的图形，1行2列布局
fig, ax= plt.subplots(2, 2, figsize=(8, 6))
random_state = 42  # 可以是任何整数
S1 = np.load('src_encoder_tsne.npy')
T1 = np.load('tgt_encoder_tsne.npy')
S1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=25).fit_transform(S1)
T1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=25).fit_transform(T1)

ax[1, 0].scatter(S1_tsne[:, 0], S1_tsne[:, 1], s=8, label='source', color='red')
ax[1, 0].scatter(T1_tsne[:, 0], T1_tsne[:, 1], s=8, label='target', color='blue')
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_xlabel('RTDETR', fontsize=10)
ax[1, 0].set_ylabel('Encoder Feature' , fontsize=12)

# 在第二个子图上绘制不同的数据
DA_S1 = np.load('src_encoder_tsne.npy')
DA_T1 = np.load('tgt_encoder_tsne.npy')
DAS1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=10).fit_transform(DA_S1)
DAT1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=10).fit_transform(DA_T1)

ax[1, 1].scatter(DAS1_tsne[:, 0], DAS1_tsne[:, 1], s=8, label='source', color='red')
ax[1, 1].scatter(DAT1_tsne[:, 0], DAT1_tsne[:, 1], s=8, label='target', color='blue')
# 不显示坐标
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])

ax[1, 1].set_xlabel('RT-DATR(Ours)', fontsize=10)


# backbone
S1 = np.load('da_src_encoder_tsne.npy')
T1 = np.load('da_tgt_encoder_tsne.npy')
S1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=100).fit_transform(S1)
T1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=100).fit_transform(T1)

ax[0, 0].scatter(S1_tsne[:, 0], S1_tsne[:, 1], s=8, label='source', color='red')
ax[0, 0].scatter(T1_tsne[:, 0], T1_tsne[:, 1], s=8, label='target', color='blue')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_ylabel('Backbone Feature', fontsize=12)



S1 = np.load('da_src_encoder_tsne.npy')
T1 = np.load('da_tgt_encoder_tsne.npy')
S1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=30).fit_transform(S1)
T1_tsne = TSNE(n_components=2, random_state=random_state, perplexity=30).fit_transform(T1)

ax[0, 1].scatter(S1_tsne[:, 0], S1_tsne[:, 1], s=8, label='source', color='red')
ax[0, 1].scatter(T1_tsne[:, 0], T1_tsne[:, 1], s=8, label='target', color='blue')
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])




# 添加图例
# plt.legend(loc='lower right')

# 调整子图之间的间距（可选）
plt.tight_layout()

# 将整个图形保存为PDF文件
plt.savefig('feature_visualization_6.pdf', bbox_inches="tight")