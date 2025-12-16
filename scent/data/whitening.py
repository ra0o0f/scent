import torch
from transformers import AutoTokenizer, AutoModel


def get_whitening_params(embeddings, n_components=None):
    mu = torch.mean(embeddings, dim=0)
    centered = embeddings - mu

    cov = torch.matmul(centered.T, centered) / centered.shape[0]

    U, S, Vh = torch.linalg.svd(cov)
    W = torch.matmul(U, torch.diag(1.0 / torch.sqrt(S + 1e-6)))

    if n_components is not None:
        W = W[:, :n_components]

    return mu, W


def extract_embeddings(data_dict, tokenizer, lm_model, device, batch_size=32):
    lm_model.eval()
    all_embeddings = []
    data_items = list(data_dict.values())

    for i in range(0, len(data_items), batch_size):
        batch_items = data_items[i : i + batch_size]

        batch_tokens = []
        for item in batch_items:
            name = item["name"]
            abstract = item.get("short_abstract")

            # <s> {name} </s> </s> {abstract} </s>
            if abstract:
                encoded = tokenizer(
                    text=name, text_pair=abstract, truncation=True, max_length=512
                )
            # <s> {name} </s>
            else:
                encoded = tokenizer(text=name, truncation=True, max_length=512)
            batch_tokens.append(encoded)

        inputs = tokenizer.pad(batch_tokens, padding=True, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = lm_model(**inputs, output_hidden_states=True)

            first_layer = outputs.hidden_states[0]
            last_layer = outputs.hidden_states[-1]
            avg_repr = (first_layer + last_layer) / 2.0

            mask = (
                inputs["attention_mask"].unsqueeze(-1).expand(avg_repr.size()).float()
            )
            sum_embeddings = torch.sum(avg_repr * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            all_embeddings.append(mean_pooled)

    return torch.cat(all_embeddings, dim=0)
