1. /ops/fusedbitnet.py中，"y = tl.math.round(y * scale)" 替换为 "y = tl.extra.cuda.libdevice.round(y * scale)"
2. 在models/hgrn_bit/modeling_hgrn_bit.py中增加 GenerationMixin 的导入，让 CausalLM 类多重继承 GenerationMixin, 移除当前自定义的 generate() 方法
3. 在generate.py的outputs中开启use_cache=False，让预填充和解码都不写 KV 缓存
4. 训练就用torchrun --nproc_per_node=2 train_tinystories.py