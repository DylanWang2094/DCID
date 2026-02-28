import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
import os
from torch.utils.data._utils.collate import default_collate
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_PATH = "./model/<your_local_model_dir>"
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)


class SarcasmDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        comment = str(row['评论内容'])
        topic = str(row['话题'])
        label = row['讽刺性[0/1]']

        combined_text = topic + " " + comment

        encoding = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        comment_encoding = self.tokenizer(
            comment,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        topic_encoding = self.tokenizer(
            topic,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        combined_encoding = self.tokenizer(
            combined_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'comment_input_ids': comment_encoding['input_ids'].squeeze(),
            'comment_attention_mask': comment_encoding['attention_mask'].squeeze(),
            'topic_input_ids': topic_encoding['input_ids'].squeeze(),
            'topic_attention_mask': topic_encoding['attention_mask'].squeeze(),
            'combined_input_ids': combined_encoding['input_ids'].squeeze(),
            'combined_attention_mask': combined_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class MultiDimensionalSemanticConflictDetector(nn.Module):
    def __init__(self, bert_model_path, hidden_dim=768, num_heads=8, output_dim=768):
        super(MultiDimensionalSemanticConflictDetector, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        self.output_dim = output_dim

        self.W_q_ra = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_ra = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_ra = nn.Linear(hidden_dim, hidden_dim)
        self.W_o_ra = nn.Linear(hidden_dim, hidden_dim)

        self.W_q_cra = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_cra = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_cra = nn.Linear(hidden_dim, hidden_dim)
        self.W_ba = nn.Linear(self.d_k, self.d_k)
        self.W_o_cra = nn.Linear(hidden_dim, hidden_dim)

        self.W_q_ia = nn.Linear(hidden_dim, hidden_dim)
        self.W_k_ia = nn.Linear(hidden_dim, hidden_dim)
        self.W_v_ia = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, comment_input_ids, comment_attention_mask, topic_input_ids, topic_attention_mask,
                combined_input_ids, combined_attention_mask):
        comment_outputs = self.bert(input_ids=comment_input_ids, attention_mask=comment_attention_mask)
        topic_outputs = self.bert(input_ids=topic_input_ids, attention_mask=topic_attention_mask)
        combined_outputs = self.bert(input_ids=combined_input_ids, attention_mask=combined_attention_mask)

        r_B = comment_outputs.last_hidden_state
        c_B = topic_outputs.last_hidden_state
        combined_B = combined_outputs.last_hidden_state

        Q_ra = self.W_q_ra(r_B)
        K_ra = self.W_k_ra(r_B)
        V_ra = self.W_v_ra(r_B)

        Q_ra = Q_ra.view(Q_ra.size(0), Q_ra.size(1), self.num_heads, self.d_k).transpose(1, 2)
        K_ra = K_ra.view(K_ra.size(0), K_ra.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V_ra = V_ra.view(V_ra.size(0), V_ra.size(1), self.num_heads, self.d_k).transpose(1, 2)

        scores_ra = torch.matmul(Q_ra, K_ra.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights_ra = torch.softmax(scores_ra, dim=-1)
        r_ra = torch.matmul(attn_weights_ra, V_ra)
        r_ra = r_ra.transpose(1, 2).contiguous().view(r_ra.size(0), r_ra.size(2), self.hidden_dim)
        r_ra = self.W_o_ra(r_ra)

        Q_cra = self.W_q_cra(r_B)
        K_cra = self.W_k_cra(c_B)
        V_cra = self.W_v_cra(c_B)

        Q_cra = Q_cra.view(Q_cra.size(0), Q_cra.size(1), self.num_heads, self.d_k).transpose(1, 2)
        K_cra = K_cra.view(K_cra.size(0), K_cra.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V_cra = V_cra.view(V_cra.size(0), V_cra.size(1), self.num_heads, self.d_k).transpose(1, 2)

        K_cra_transformed = self.W_ba(K_cra)
        scores_cra = torch.matmul(Q_cra, K_cra_transformed.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights_cra = torch.softmax(scores_cra, dim=-1)
        r_cra = torch.matmul(attn_weights_cra, V_cra)
        r_cra = r_cra.transpose(1, 2).contiguous().view(r_cra.size(0), r_cra.size(2), self.hidden_dim)
        r_cra = self.W_o_cra(r_cra)

        r_avg_ra = r_ra.mean(dim=1, keepdim=True)
        r_avg_cra = r_cra.mean(dim=1, keepdim=True)
        X_IA = torch.cat([r_avg_ra, r_avg_cra], dim=1)

        Q_IA = self.W_q_ia(X_IA)
        K_IA = self.W_k_ia(X_IA)
        V_IA = self.W_v_ia(X_IA)

        scores_IA = torch.matmul(Q_IA, K_IA.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attn_weights_IA = torch.softmax(scores_IA, dim=-1)
        X_IA_prime = torch.matmul(attn_weights_IA, V_IA)

        r_ra_prime, r_cra_prime = X_IA_prime[:, 0, :], X_IA_prime[:, 1, :]
        contrast_feature = torch.cat([r_ra_prime, r_cra_prime], dim=-1)
        return contrast_feature


class EmotionalEvolutionTrajectoryTracker(nn.Module):
    def __init__(self, bert_model, hidden_dim=256, num_filters=256, filter_sizes=[3, 5, 7], num_hops=3):
        super().__init__()
        self.bert = bert_model
        self.bert_dim = 768
        self.hidden_dim = hidden_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_hops = num_hops

        self.conv1 = nn.Conv1d(self.bert_dim, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.bert_dim, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(self.bert_dim, num_filters, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.linear_cnn = nn.Linear(num_filters * len(filter_sizes), hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.bilstm = nn.LSTM(self.bert_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.linear_bilstm = nn.Linear(hidden_dim, hidden_dim)

        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        word_embeddings = bert_outputs.last_hidden_state

        x = word_embeddings.permute(0, 2, 1)
        conv1_output = self.bn1(F.relu(self.conv1(x))).max(dim=2)[0]
        conv2_output = self.bn2(F.relu(self.conv2(x))).max(dim=2)[0]
        conv3_output = self.bn3(F.relu(self.conv3(x))).max(dim=2)[0]
        cnn_out = torch.cat((conv1_output, conv2_output, conv3_output), dim=1)
        cnn_out = self.sigmoid(self.linear_cnn(cnn_out))

        batch_size = word_embeddings.size(0)
        h0 = self.linear_bilstm(cnn_out).view(batch_size, 2, self.hidden_dim // 2)
        h0 = h0.permute(1, 0, 2).contiguous()
        c0 = torch.zeros(2, batch_size, self.hidden_dim // 2, device=h0.device)
        bilstm_out, _ = self.bilstm(word_embeddings, (h0, c0))

        H = bilstm_out
        q_c = torch.randn(batch_size, self.hidden_dim, device=H.device)
        for _ in range(self.num_hops):
            h_prime = self.tanh(self.attn_linear(H))
            attn_scores = torch.bmm(h_prime, q_c.unsqueeze(2)).squeeze(2)
            attn_weights = F.softmax(attn_scores, dim=1)
            q_new = torch.bmm(attn_weights.unsqueeze(1), H).squeeze(1)
            q_c = q_c + q_new

        f_average = torch.mean(H, dim=1)
        H_context = torch.cat([f_average, q_new], dim=1)
        return H_context


class SarcasmDetectionModule(nn.Module):
    def __init__(self, input_dims, hidden_dim=256, num_heads=8, fc_layers=2):
        super(SarcasmDetectionModule, self).__init__()
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.input_dims = input_dims
        self.fc_layers = fc_layers

        self.feature_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        self.word_emb_projection = nn.Linear(768, hidden_dim)

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        total_dim = 256 + 768 + sum(input_dims)
        fc_modules = []

        if fc_layers == 2:
            fc_modules.append(nn.Linear(total_dim, hidden_dim))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(0.3))
            fc_modules.append(nn.Linear(hidden_dim, 2))
        elif fc_layers == 3:
            fc_modules.append(nn.Linear(total_dim, hidden_dim * 2))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(0.3))
            fc_modules.append(nn.Linear(hidden_dim * 2, hidden_dim))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(0.3))
            fc_modules.append(nn.Linear(hidden_dim, 2))
        elif fc_layers == 4:
            fc_modules.append(nn.Linear(total_dim, hidden_dim * 3))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(0.3))
            fc_modules.append(nn.Linear(hidden_dim * 3, hidden_dim * 2))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(0.3))
            fc_modules.append(nn.Linear(hidden_dim * 2, hidden_dim))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(0.3))
            fc_modules.append(nn.Linear(hidden_dim, 2))

        self.fc_network = nn.Sequential(*fc_modules)

    def forward(self, features, word_embeddings):
        batch_size = word_embeddings.size(0)

        feature_vectors = [
            proj(f) for proj, f in zip(self.feature_projections, features)
        ]
        word_emb_avg = word_embeddings.mean(dim=1)
        word_emb_proj = self.word_emb_projection(word_emb_avg).unsqueeze(1)
        feature_vectors = torch.cat([torch.stack(feature_vectors, dim=1), word_emb_proj], dim=1)

        Q = self.W_q(feature_vectors)
        K = self.W_k(feature_vectors)
        V = self.W_v(feature_vectors)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        multihead_output = self.W_o(attn_output)
        multihead_avg = multihead_output.mean(dim=1)

        f_final = torch.cat([multihead_avg, word_emb_avg] + features, dim=-1)

        logits = self.fc_network(f_final)
        return logits


class SarcasmModel(nn.Module):
    def __init__(self, bert_model_path, lstm_hidden_dim=256, num_filters=256, fc_layers=2):
        super(SarcasmModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path).to(device)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.module3 = MultiDimensionalSemanticConflictDetector(bert_model_path)
        self.module4 = EmotionalEvolutionTrajectoryTracker(self.bert, hidden_dim=lstm_hidden_dim,
                                                           num_filters=num_filters)

        input_dims = [1536, lstm_hidden_dim * 2]
        self.module6 = SarcasmDetectionModule(input_dims, hidden_dim=256, num_heads=8, fc_layers=fc_layers)

    def forward(self, input_ids, attention_mask, comment_input_ids, comment_attention_mask,
                topic_input_ids, topic_attention_mask, combined_input_ids, combined_attention_mask):
        feature3 = self.module3(comment_input_ids, comment_attention_mask, topic_input_ids, topic_attention_mask,
                                combined_input_ids, combined_attention_mask)
        feature4 = self.module4(input_ids, attention_mask)
        word_embeddings = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        features = [feature3, feature4]
        logits = self.module6(features, word_embeddings)
        return logits


def custom_collate_fn(batch):
    tensor_keys = [
        'input_ids', 'attention_mask', 'comment_input_ids', 'comment_attention_mask',
        'topic_input_ids', 'topic_attention_mask', 'combined_input_ids', 'combined_attention_mask',
        'label'
    ]
    tensor_batch = [
        {key: sample[key] for key in tensor_keys}
        for sample in batch
    ]
    collated_tensor_batch = default_collate(tensor_batch)
    return collated_tensor_batch


def load_data(csv_path, tokenizer, batch_size=16, random_seed=41):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')

    if df['讽刺性[0/1]'].isna().any() or not all(df['讽刺性[0/1]'].isin([0, 1])):
        print("警告：数据集中存在无效或缺失的标签，已过滤")
        df = df[df['讽刺性[0/1]'].isin([0, 1])].dropna()

    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=random_seed)
    val_df, test_df = train_test_split(temp_df, test_size=2 / 3, random_state=random_seed)

    train_dataset = SarcasmDataset(train_df, tokenizer)
    val_dataset = SarcasmDataset(val_df, tokenizer)
    test_dataset = SarcasmDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader


def calculate_metrics(model, dataloader, threshold=0.5, search_threshold=False):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            comment_input_ids = batch['comment_input_ids'].to(device)
            comment_attention_mask = batch['comment_attention_mask'].to(device)
            topic_input_ids = batch['topic_input_ids'].to(device)
            topic_attention_mask = batch['topic_attention_mask'].to(device)
            combined_input_ids = batch['combined_input_ids'].to(device)
            combined_attention_mask = batch['combined_attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, comment_input_ids, comment_attention_mask,
                            topic_input_ids, topic_attention_mask, combined_input_ids, combined_attention_mask)
            probs = torch.softmax(outputs, dim=-1)[:, 1]
            y_pred.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    best_threshold = threshold
    if search_threshold:
        best_f1 = -1
        for t in np.arange(0.3, 0.7, 0.02):
            f1 = f1_score(y_true, (y_pred > t).astype(int))
            if f1 > best_f1:
                best_f1, best_threshold = f1, t

    y_pred_binary = (y_pred > best_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, auc, best_threshold


def train_model(model, train_loader, val_loader, epochs=50, lr=2e-5):
    bert_params = list(model.bert.parameters()) + list(model.module3.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]

    optimizer = optim.AdamW([
        {'params': bert_params, 'lr': lr * 0.1},
        {'params': other_params, 'lr': lr}
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    best_f1 = -1
    best_metrics = None
    best_threshold = 0.5
    patience = 5
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            comment_input_ids = batch['comment_input_ids'].to(device)
            comment_attention_mask = batch['comment_attention_mask'].to(device)
            topic_input_ids = batch['topic_input_ids'].to(device)
            topic_attention_mask = batch['topic_attention_mask'].to(device)
            combined_input_ids = batch['combined_input_ids'].to(device)
            combined_attention_mask = batch['combined_attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, comment_input_ids, comment_attention_mask,
                           topic_input_ids, topic_attention_mask, combined_input_ids, combined_attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, 训练损失: {avg_loss:.4f}")

        val_accuracy, val_precision, val_recall, val_f1, val_auc, val_threshold = calculate_metrics(model, val_loader,
                                                                                                    threshold=0.5,
                                                                                                    search_threshold=True)
        print(
            f"验证集 - 准确率: {val_accuracy:.4f}, 精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = (val_accuracy, val_precision, val_recall, val_f1, val_auc)
            best_threshold = val_threshold
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停触发，第 {epoch + 1} 个epoch后停止")
                break

    print("\n最佳验证集指标:")
    print(
        f"准确率: {best_metrics[0]:.4f}, 精确率: {best_metrics[1]:.4f}, 召回率: {best_metrics[2]:.4f}, F1: {best_metrics[3]:.4f}, AUC: {best_metrics[4]:.4f}")
    return model, best_metrics, best_threshold


def evaluate_model(model, test_loader, best_threshold):
    model.eval()
    accuracy, precision, recall, f1, auc, _ = calculate_metrics(model, test_loader, threshold=best_threshold,
                                                                search_threshold=False)
    print("\n测试集评估结果:")
    print(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return accuracy, precision, recall, f1, auc


def main():
    csv_path = "data.csv"

    random_seeds = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

    results_columns = [
        'experiment_id', 'random_seed',
        'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_auc',
        'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_auc',
        'best_threshold', 'training_time'
    ]
    results_df = pd.DataFrame(columns=results_columns)

    exp_count = 0
    total_experiments = len(random_seeds)

    print(f"开始进行{total_experiments}组实验...")
    print(f"  随机种子: {random_seeds}")

    for seed in random_seeds:
        exp_count += 1
        print(f"\n{'=' * 80}")
        print(f"实验 {exp_count}/{total_experiments}")
        print(f"参数: seed={seed}")
        print(f"{'=' * 80}")

        start_time = datetime.now()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_loader, val_loader, test_loader = load_data(csv_path, tokenizer, batch_size=16,
                                                          random_seed=seed)

        model = SarcasmModel(BERT_PATH, lstm_hidden_dim=320,
                             num_filters=384, fc_layers=4).to(device)

        trained_model, best_val_metrics, best_threshold = train_model(model, train_loader, val_loader)

        test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate_model(trained_model,
                                                                                       test_loader,
                                                                                       best_threshold)

        training_time = (datetime.now() - start_time).total_seconds()

        new_row = {
            'experiment_id': exp_count,
            'random_seed': seed,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'val_accuracy': best_val_metrics[0],
            'val_precision': best_val_metrics[1],
            'val_recall': best_val_metrics[2],
            'val_f1': best_val_metrics[3],
            'val_auc': best_val_metrics[4],
            'best_threshold': best_threshold,
            'training_time': training_time
        }

        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        results_df.to_csv('experiment_results.csv', index=False)

        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print(f"实验总结")
    print(f"{'=' * 80}")
    print(f"总共完成{exp_count}组实验")

    totalnum_experiments = results_df[results_df['test_f1'] >= 0]
    if not totalnum_experiments.empty:
        print(f"\n总实验数量: {len(totalnum_experiments)}")
        print("\n实验详情:")
        print(totalnum_experiments[['experiment_id', 'random_seed', 'test_f1']].to_string())

    print(f"\n结果已保存到: experiment_results.csv")
    print(f"CSV文件包含{len(results_df)}组实验的完整结果")


if __name__ == "__main__":
    main()
