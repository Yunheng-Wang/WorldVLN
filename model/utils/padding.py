import torch


def padding_t5_text(video_text_embedding, text_len):
    padded_embeddings = []
    for emb in video_text_embedding:
        if emb.shape[0] <= text_len:
            padded = torch.cat([emb, emb.new_zeros(text_len - emb.shape[0], emb.shape[1])])
        else:
            padded = emb[:text_len]
        padded_embeddings.append(padded)
    
    return torch.stack(padded_embeddings, dim=0)