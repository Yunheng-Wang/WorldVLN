import torch


def padding_t5_text_muilt(video_text_embedding, text_len):
    # video_text_embedding: 输入 [embed1, embed2, ...]
    padded_embeddings = []
    for emb in video_text_embedding:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    
    return torch.stack(padded_embeddings, dim=0)


def padding_t5_text_single(video_text_embedding, text_len):
    # video_text_embedding: 输入 embed1
    if video_text_embedding.shape[0] <= text_len:
        padded = torch.cat([video_text_embedding, video_text_embedding.new_zeros(text_len - video_text_embedding.shape[0], video_text_embedding.shape[1])])
    else:
        padded = video_text_embedding[:text_len]
    return padded