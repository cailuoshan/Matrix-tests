import torch
import numpy as np

def generate_random_matrix(shape, device='cpu'):
    matrix = torch.randn(shape)
    return matrix.to(device)


if __name__ == "__main__":
    shape1 = (4096, 4096)
    shape2 = (4096, 1024)
    shape3 = (4096, 1024)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成随机矩阵(默认数据类型为float32)
    matrixA = generate_random_matrix(shape1, device)
    matrixB = generate_random_matrix(shape2, device)
    matrixC = generate_random_matrix(shape3, device)
    
    # 计算矩阵乘结果
    resultC = torch.add(torch.matmul(matrixA, matrixB), matrixC)

    with open('matrixA.bin', 'wb') as f: f.write(matrixA.numpy().tobytes())
    with open('matrixB.bin', 'wb') as f: f.write(matrixB.numpy().tobytes())
    with open('matrixC.bin', 'wb') as f: f.write(matrixC.numpy().tobytes())
    with open('resultC.bin', 'wb') as f: f.write(resultC.numpy().tobytes())