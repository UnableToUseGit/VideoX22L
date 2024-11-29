import torch
import torch.nn.functional as F
import pdb

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 生成测试数据
batch_size = 2  # 批次大小
num_heads = 4  # 多头注意力中的头数量
seq_len_q = 5  # 查询序列长度
seq_len_k = 7  # 键值序列长度
dim_per_head = 8  # 每个头的维度

# 创建随机张量 (query, key, value)
query_states = torch.rand(batch_size, num_heads, seq_len_q, dim_per_head)
key_states = torch.rand(batch_size, num_heads, seq_len_k, dim_per_head)
value_states = torch.rand(batch_size, num_heads, seq_len_k, dim_per_head)

# 可选的注意力掩码
attention_mask = torch.zeros(batch_size, 1, seq_len_q, seq_len_k)
attention_mask[:, :, 3:, :] = -float('inf')  # 对某些位置施加掩码（示例）

# 是否为因果注意力
is_causal = True

# 设置注意力dropout
attention_dropout = 0.1

# 测试函数
pdb.set_trace()
output = F.scaled_dot_product_attention(
    query_states,
    key_states,
    value_states,
    attn_mask=attention_mask,
    dropout_p=attention_dropout if torch.is_training() else 0.0,
    is_causal=is_causal and attention_mask is None and seq_len_q > 1
)

# 输出结果
print("Scaled Dot Product Attention Output:")
print(output)
