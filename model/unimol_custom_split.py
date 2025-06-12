import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unimol_tools import MolTrain, MolPredict
from data.data_load import load_all_bbbp_data
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


def prepare_combined_data():
    """
    合并训练集和验证集，并为每个样本添加组标识符
    """
    print("正在加载BBBP数据集...")
    
    # 加载训练集、验证集和测试集
    train_data, valid_data, test_data = load_all_bbbp_data()
    
    print(f"训练集大小: {len(train_data['atoms'])}")
    print(f"验证集大小: {len(valid_data['atoms'])}")
    print(f"测试集大小: {len(test_data['atoms'])}")
    
    # 合并训练集和验证集
    combined_atoms = train_data['atoms'] + valid_data['atoms']
    combined_coordinates = train_data['coordinates'] + valid_data['coordinates']
    combined_targets = [int(target[0]) for target in train_data['target']] + \
                      [int(target[0]) for target in valid_data['target']]
    
    # 创建组标识符：0表示训练集，1表示验证集
    # 这样在2折交叉验证中，fold 0 将使用训练集作为训练，验证集作为验证
    # fold 1 将相反（但我们只使用fold 0的结果）
    groups = [0] * len(train_data['atoms']) + [1] * len(valid_data['atoms'])
    
    combined_data = {
        'atoms': combined_atoms,
        'coordinates': combined_coordinates,
        'target': combined_targets,
        'group': groups 
    }
    
    return combined_data, test_data


def train_with_custom_split():
    """
    只使用训练集进行训练，然后在验证集和测试集上分别评估
    """
    train_data, valid_data, test_data = load_all_bbbp_data()
    
    print(f"训练集大小: {len(train_data['atoms'])}")
    print(f"验证集大小: {len(valid_data['atoms'])}")
    print(f"测试集大小: {len(test_data['atoms'])}")
    
    if 'smiles' in train_data:
        print("检测到SMILES数据，将使用真实的SMILES信息")
        print(f"样本SMILES: {train_data['smiles'][0]}")
    
    # 准备训练数据（只使用训练集）
    train_data_formatted = {
        'atoms': train_data['atoms'],
        'coordinates': train_data['coordinates'],
        'target': [int(target[0]) for target in train_data['target']]
    }
    
    # 如果有SMILES数据，添加到训练数据中（使用大写SMILES作为键名）
    if 'smiles' in train_data:
        train_data_formatted['SMILES'] = train_data['smiles']
    
    print("正在初始化UniMol训练器（只使用训练集）...")
    
    # 初始化训练器，使用kfold=1表示不进行交叉验证
    clf = MolTrain(
        task='classification',
        data_type='molecule',
        epochs=40,
        batch_size=128,
        metrics='auc',
        learning_rate=4e-4,
        kfold=1,  # 不进行交叉验证，使用所有训练数据进行训练
        save_path='./exp/bbbp_model_train_only'
    )
    
    print("开始训练模型（只使用训练集）...")
    
    try:
        # 训练模型
        clf.fit(data=train_data_formatted)
        print("模型训练完成!")
        
        return clf, valid_data, test_data
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def evaluate_on_dataset(model_path, data, dataset_name):
    """
    在指定数据集上评估模型
    """
    print(f"\n正在在{dataset_name}上评估模型...")
    
    # 准备数据格式
    eval_data_formatted = {
        'atoms': data['atoms'],
        'coordinates': data['coordinates']
    }
    
    try:
        # 初始化预测器
        predictor = MolPredict(load_model=model_path)
        
        # 进行预测
        predictions = predictor.predict(data=eval_data_formatted)
        
        # 计算评估指标
        if 'target' in data and len(data['target']) > 0:
            labels = [int(target[0]) for target in data['target']]
            pred_array = np.array(predictions)
            
            print(f"{dataset_name}预测结果形状: {pred_array.shape}")
            print(f"{dataset_name}预测值范围: [{np.min(pred_array):.4f}, {np.max(pred_array):.4f}]")
            
            # 处理预测结果
            if pred_array.ndim > 1 and pred_array.shape[1] > 1:
                probabilities = pred_array[:, 1]
                predicted_classes = np.argmax(pred_array, axis=1)
                print(f"{dataset_name}检测到多类输出格式")
            else:
                pred_array = pred_array.flatten()
                if np.max(pred_array) <= 1.0 and np.min(pred_array) >= 0.0:
                    probabilities = pred_array
                    print(f"{dataset_name}检测到概率输出格式")
                else:
                    probabilities = 1 / (1 + np.exp(-pred_array))
                    print(f"{dataset_name}检测到logits输出格式，已转换为概率")
                
                predicted_classes = (probabilities > 0.5).astype(int)
            
            # 计算指标
            accuracy = accuracy_score(labels, predicted_classes)
            roc_auc = roc_auc_score(labels, probabilities)
            
            print(f"\n{dataset_name}评估结果：")
            print(f"准确率 (Accuracy): {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"详细分类报告：")
            print(classification_report(labels, predicted_classes, 
                                      target_names=['非BBB穿透', 'BBB穿透']))
            
            return {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'predictions': predictions,
                'probabilities': probabilities,
                'labels': labels
            }
        else:
            print(f"{dataset_name}没有标签，无法计算评估指标")
            return predictions
            
    except Exception as e:
        print(f"在{dataset_name}上评估时出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    主函数：训练模型并在验证集和测试集上分别评估
    """
    print("开始BBBP数据集训练（仅使用训练集）")
    print("=" * 60)
    
    # 训练模型
    clf, valid_data, test_data = train_with_custom_split()
    
    if clf is not None and valid_data is not None and test_data is not None:
        print("训练成功完成！")
        
        # 在验证集上评估
        valid_results = evaluate_on_dataset('./exp/bbbp_model_train_only', valid_data, '验证集')
        
        # 在测试集上评估
        test_results = evaluate_on_dataset('./exp/bbbp_model_train_only', test_data, '测试集')
        
        # 总结结果
        print("\n" + "=" * 60)
        print("最终评估结果总结")
        print("=" * 60)
        
        if valid_results and isinstance(valid_results, dict):
            print(f"验证集性能:")
            print(f"  - 准确率: {valid_results['accuracy']:.4f}")
            print(f"  - ROC-AUC: {valid_results['roc_auc']:.4f}")
        
        if test_results and isinstance(test_results, dict):
            print(f"测试集性能:")
            print(f"  - 准确率: {test_results['accuracy']:.4f}")
            print(f"  - ROC-AUC: {test_results['roc_auc']:.4f}")
        
        print("\n模型训练和评估完成！")
        return {
            'model': clf,
            'validation_results': valid_results,
            'test_results': test_results
        }
    else:
        print("训练失败，请检查错误信息")
        return None


if __name__ == "__main__":
    main() 