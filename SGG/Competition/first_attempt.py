import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 加载和预处理数据
def load_and_preprocess_data(filepath):
    # 读取CSV文件
    df = pd.read_csv(filepath)

    # 分离特征和标签
    X = df[['Feature_1', 'Feature_2', 'Feature_3']]
    y = df['Label']

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# 2. 模型训练和评估函数
def train_and_evaluate_models(X, y):
    # 划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    # 初始化模型
    models = {
        'Random Forest': RandomForestClassifier(n_estimators = 100, random_state = 42),
        'BP Neural Network': MLPClassifier(
            hidden_layer_sizes = (100, 50),  # 两个隐藏层，分别有100和50个神经元
            activation = 'relu',  # ReLU激活函数
            solver = 'adam',  # Adam优化器
            max_iter = 1000,  # 最大迭代次数
            random_state = 42
        )
    }

    # 训练和评估每个模型
    results = {}
    for name, model in models.items():
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv = 5)

        # 在完整训练集上训练模型
        model.fit(X_train, y_train)

        # 在验证集上评估
        val_score = model.score(X_val, y_val)

        # 存储结果
        results[name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'val_score': val_score,
            'model': model
        }

        print(f"\n{name} Results:")
        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Validation score: {val_score:.4f}")

        # 打印详细分类报告
        y_pred = model.predict(X_val)
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))

        # 绘制混淆矩阵
        plt.figure(figsize = (8, 6))
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    return results


# 3. 主函数
def main():
    # 加载数据
    print("Loading and preprocessing data...")
    X, y, scaler = load_and_preprocess_data('train_dataset.csv')

    # 训练和评估模型
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X, y)

    # 选择最佳模型
    best_model_name = max(results, key = lambda k: results[k]['val_score'])
    best_model = results[best_model_name]['model']
    print(f"\nBest performing model: {best_model_name}")
    print(f"Validation score: {results[best_model_name]['val_score']:.4f}")

    # 保存最佳模型
    import joblib
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nBest model and scaler saved to files.")

    return best_model, scaler


if __name__ == "__main__":
    best_model, scaler = main()
