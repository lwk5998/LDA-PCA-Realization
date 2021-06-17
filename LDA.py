import pandas as pd
# 创建标签
feature_dict = {i:label for i,label in zip(
                range(4),
                  ('sepal length in cm',
                  'sepal width in cm',
                  'petal length in cm',
                  'petal width in cm', ))}
# 导入数据
df = pd.io.parsers.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',',
    )
# 修改列名
df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end
# 展示
df.tail()

from sklearn.preprocessing import LabelEncoder

X = df[['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm']].values
y = df['class label'].values
# 实例化
enc = LabelEncoder()
# 执行操作
label_encoder = enc.fit(y)
y = label_encoder.transform(y) + 1

import numpy as np
np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))

# 构造矩阵 4*4
S_W = np.zeros((4,4))
for cl,mv in zip(range(1,4), mean_vectors):
    class_sc_mat = np.zeros((4,4))                  # 每个类的散布矩阵
    for row in X[y == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # 生成列向量
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # 求和
print('类内散布矩阵：\n', S_W)

overall_mean = np.mean(X, axis=0)
S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):  
    n = X[y==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(4,1)
    overall_mean = overall_mean.reshape(4,1)
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
print('类间散布矩阵：\n', S_B)

# 拿到特征值和特征向量
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# 展示特征值与特征向量
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)   
    print('\nEigenvector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigenvalue {:}: {:.2e}'.format(i+1, eig_vals[i].real))

# 列出（特征值，特征向量）元组
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# 将（特征值，特征向量）元组从高到低排序
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
# 通过降低特征值来直观地确认列表已正确排序
print('降阶特征值:\n')
for i in eig_pairs:
    print(i[0])

print('差异百分比:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('特征值 {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))

W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('矩阵 W:\n', W.real)


# 降维
X_lda = X.dot(W) # [150,4]*[4,2] → [150,2]
# 判断结果
assert X_lda.shape == (150,2), "The matrix is not 150x2 dimensional."


from matplotlib import pyplot as plt

label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

def plot_step_lda():
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):
        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')
    plt.grid()
    plt.tight_layout
    plt.show()
# 执行
plot_step_lda()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# LDA 实例化 降低为2维
sklearn_lda = LDA(n_components=2)
# 执行
X_lda_sklearn = sklearn_lda.fit_transform(X, y)


