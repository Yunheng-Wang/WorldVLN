def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("="*50)

    # 判断并打印 Video Model 参数量
    if hasattr(model, 'video_model') and hasattr(model.video_model, 'parameters'):
        video_params = sum(p.numel() for p in model.video_model.parameters())
        print(f"Video Model 参数量: {video_params / 1e9:.2f} B")

    # 判断并打印 Video Model Text Encoder 参数量
    if hasattr(model, 'video_model') and hasattr(model.video_model, 't5') and hasattr(model.video_model.t5, 'model'):
        t5_params = sum(p.numel() for p in model.video_model.t5.model.parameters())
        print(f"Video Model Text Encoder 参数量: {t5_params / 1e9:.2f} B")

    # 判断并打印 Action Model 参数量
    if hasattr(model, 'action_model') and hasattr(model.action_model, 'parameters'):
        action_params = sum(p.numel() for p in model.action_model.parameters())
        print(f"Action Model 参数量: {action_params / 1e9:.2f} B")

    # 判断并打印 Understand Model 参数量
    if hasattr(model, 'understand_model') and hasattr(model.understand_model, 'parameters'):
        understand_params = sum(p.numel() for p in model.understand_model.parameters())
        print(f"Understand Model 参数量: {understand_params / 1e9:.2f} B")

    # 判断并打印 Understand Model VLM 参数量
    if hasattr(model, 'understand_model') and hasattr(model.understand_model, 'vlm_model') and hasattr(model.understand_model.vlm_model, 'parameters'):
        VLM_params = sum(p.numel() for p in model.understand_model.vlm_model.parameters())
        print(f"Understand Model VLM 参数量: {VLM_params / 1e9:.2f} B")

    print("="*50)