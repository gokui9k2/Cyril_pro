import torch
import torch.nn as nn
import torch.nn.functional as F
    
class PETNN_Cell(nn.Module):
    def __init__(self, input_size, hidden_size, T_init=1.0, hard_switch=True, Rt_bias=1.5, Zt_bias=1.0, dropout=0.3):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Rt_bias = Rt_bias
        self.Zt_bias = Zt_bias
        self.dropout_rate = dropout
        self.T_init = T_init
        self.hard_switch = hard_switch

        self.WIt = nn.Linear(input_size, hidden_size)
        self.WZt = nn.Linear(input_size + hidden_size, hidden_size)
        self.WRt = nn.Linear(input_size, hidden_size)
        self.WZc = nn.Linear(input_size + hidden_size, hidden_size)
        self.WZw = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wh = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)

        self.ln_h_input = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    if module == self.WRt:
                        nn.init.constant_(module.bias, self.Rt_bias)
                    elif module == self.WZt:
                        nn.init.constant_(module.bias, self.Zt_bias)

    def update_step(self, X_t, S_prev, C_prev, T_prev):

        It = self.WIt(X_t)
        Rt = F.softplus(self.WRt(X_t))
        Rt = torch.clamp(Rt, max=3.0)

        concat_input = torch.cat([X_t, S_prev], dim=-1)
        Zt = self.WZt(concat_input)
        Zc = self.WZc(concat_input)
        Zw = self.WZw(concat_input)

        T_new = Rt * torch.sigmoid((T_prev + Zt)) - 1
        num_resets = 0

        if self.hard_switch:
            m = (T_new < 0).float()
            num_resets = (T_new < 0).sum().item()
            T_new = T_new.clamp(min=0)
        else:
            m = torch.sigmoid(T_new)

        C_new = (1 - m) * C_prev + m * It + Zc
        C_modulated = (1 - m) * C_prev

        h_input = torch.cat([X_t, S_prev, C_modulated], dim=-1)
        h_state = self.Wh(h_input)
        h_state = self.ln_h_input(h_state)

        state_update = (1 - Zw) * S_prev + Zw * h_state
        hidden_st = self.dropout(state_update)
        S_new = torch.tanh(hidden_st)

        return S_new, C_new, T_new,hidden_st ,num_resets

class PETNN_Masked_Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, patch_size=6, stride=4,
                 T_init=1.0, hard_switch=True, Rt_bias=1.5, Zt_bias=1.0, dropout=0.3):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride

        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size,kernel_size=patch_size,stride=stride,padding=0)

        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size)

        self.cell = PETNN_Cell(hidden_size, hidden_size,T_init, hard_switch,Rt_bias, Zt_bias, dropout)


    def forward(self, x, mask):

        mask_expanded = mask.repeat_interleave(self.patch_size, dim=1) # (B, 96)
        mask_expanded = mask_expanded.unsqueeze(-1)                    # (B, 96, 1)

        x = torch.cat([x, mask_expanded], dim=-1)

        B, L, F = x.shape
        device = x.device

        x_conv = x.transpose(1, 2)             # (B, F+1, L)
        x_encoded = self.conv(x_conv)          # (B, hidden_size, L')
        x_encoded = x_encoded.transpose(1, 2)  # (B, L', hidden_size)

        x_encoded = self.gelu(x_encoded)
        x_encoded = self.ln(x_encoded)

        S_prev = torch.zeros(B, self.cell.hidden_size, device=device)
        T_prev = torch.full((B, self.cell.hidden_size), self.cell.T_init, device=device)
        C_prev = torch.zeros(B, self.cell.hidden_size, device=device)

        outputs = []
        total_resets = 0

        for t in range(x_encoded.size(1)):
            X_t = x_encoded[:, t, :]
            S_prev, C_prev, T_prev, h_t, resets = self.cell.update_step(X_t, S_prev, C_prev, T_prev)
            outputs.append(h_t)

            total_resets += resets
        outputs = torch.stack(outputs, dim=1)  # (B, L', hidden_size)

        return outputs, total_resets

class PETNN_DirectHead(nn.Module):
    def __init__(self, input_dim, forecast_horizon, output_dim, dropout=0.3):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout), 
            nn.Linear(input_dim, forecast_horizon * output_dim)
        )

    def forward(self, encoder_outputs):
        B = encoder_outputs.shape[0]
        prediction_flat = self.head(encoder_outputs)
        prediction = prediction_flat.view(B, self.forecast_horizon, self.output_dim)
        return prediction
    
class PETNN_Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, patch_size=6, stride=4,
                 T_init=1.0, hard_switch=True, Rt_bias=1.5, Zt_bias=1.0, dropout=0.3):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride

        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size,kernel_size=patch_size,stride=stride,padding=0)

        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size)

        self.cell = PETNN_Cell(hidden_size, hidden_size,T_init, hard_switch,Rt_bias, Zt_bias, dropout)


    def forward(self, x):

        B, L, F = x.shape
        device = x.device

        x_conv = x.transpose(1, 2)             # (B, F+1, L)
        x_encoded = self.conv(x_conv)          # (B, hidden_size, L')
        x_encoded = x_encoded.transpose(1, 2)  # (B, L', hidden_size)

        x_encoded = self.gelu(x_encoded)
        x_encoded = self.ln(x_encoded)

        S_prev = torch.zeros(B, self.cell.hidden_size, device=device)
        T_prev = torch.full((B, self.cell.hidden_size), self.cell.T_init, device=device)
        C_prev = torch.zeros(B, self.cell.hidden_size, device=device)

        outputs = []
        total_resets = 0

        for t in range(x_encoded.size(1)):
            X_t = x_encoded[:, t, :]
            S_prev, C_prev, T_prev, h_t, resets = self.cell.update_step(X_t, S_prev, C_prev, T_prev)
            outputs.append(h_t)

            total_resets += resets
        outputs = torch.stack(outputs, dim=1)  # (B, L', hidden_size)

        return outputs, total_resets

class LinearHead(nn.Module):
    def __init__(self, hidden_size, patch_size, num_features):
        super().__init__()

        self.pred = nn.Linear(hidden_size, patch_size * num_features)
        self.num_features = num_features 

    def forward(self, x):
        batch_size, num_patches, hidden = x.shape

        x = self.pred(x)

        x = x.view(batch_size, -1, self.num_features)
        return x
