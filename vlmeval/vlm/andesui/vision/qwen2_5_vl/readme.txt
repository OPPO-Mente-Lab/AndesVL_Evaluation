相比qwen2vl-vit的修改：
1. 新增qwen2rmsnorm并替换layernorm
2. mlp被替换（注意一个维度的变化）
3. flash attention的apply_rotary
4. vision rope的theta这里代码内就为10000了，不支持外部修改（不是之前的和llm共享了）
5. 多个一个fullatt_block_indexes中指定的，在哪些层使用全注意力（其他层使用局部注意力）。