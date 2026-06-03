import numpy as np
import torch
import torch.nn as nn


def _mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))

class PhysicsSiameseNet(nn.Module):
    def __init__(self, emb_dim):
        super(PhysicsSiameseNet, self).__init__()
        self.width = 128
        
        # Input is now just (x1, x2) concatenated: emb_dim * 2
        self.delta_net = nn.Sequential(
            nn.Linear(emb_dim * 2, self.width),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 1)
        )
        
        self.interaction_net = nn.Sequential(
            nn.Linear(emb_dim * 2, self.width),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            nn.Linear(self.width, 1) 
        )

    def forward(self, x1, x2):
        prod = x1 * x2
        diff_12 = x1 - x2
        diff_21 = x2 - x1
        context_12 = torch.cat((diff_12, prod), dim=1)
        context_21 = torch.cat((diff_21, prod), dim=1)

        d1_raw = self.delta_net(context_12)
        d2_raw = self.delta_net(context_21)

        delta1 = torch.nn.functional.softplus(d1_raw) + 1.0
        delta2 = torch.nn.functional.softplus(d2_raw) + 1.0

        w_a = self.interaction_net(context_12)
        w_b = self.interaction_net(context_21)
        W = ((w_a + w_b) / 2.0) * 10000.0

        return delta1.view(-1), delta2.view(-1), W.view(-1)


def m_physics_loss(d1, d2, W, T1, T2, r, Tm_target, R=8.314):
    eps = 1e-8
    r_c = torch.clamp(r, min=eps, max=1.0 - eps)
    T_ref = (T1 + T2) / 2.0
    ln_gamma1 = (W / (R * T_ref)) * (1 - r_c)**2
    ln_gamma2 = (W / (R * T_ref)) * (r_c)**2
    ln_a1 = torch.log(r_c) + ln_gamma1
    ln_a2 = torch.log(1 - r_c) + ln_gamma2
    denom1 = torch.clamp(1.0 - (R / d1) * ln_a1, min=0.1)
    denom2 = torch.clamp(1.0 - (R / d2) * ln_a2, min=0.1)
    Tm_pred = torch.max(T1 / denom1, T2 / denom2)
    return nn.L1Loss()(Tm_pred, Tm_target)


def get_model_mae(model, X1, X2, T1, T2, frac, Tm_actual, R=8.314):
    model.eval()
    with torch.no_grad():
        X1_t = torch.tensor(np.array(X1).astype(np.float32))
        X2_t = torch.tensor(np.array(X2).astype(np.float32))
        T1_t = torch.tensor(np.array(T1).astype(np.float32))
        T2_t = torch.tensor(np.array(T2).astype(np.float32))
        r_t  = torch.tensor(np.array(frac).astype(np.float32))
        d1, d2, W = model(X1_t, X2_t)
        eps = 1e-8
        r_c = torch.clamp(r_t, min=eps, max=1.0 - eps)
        T_ref = (T1_t + T2_t) / 2.0
        ln_gamma1 = (W / (R * T_ref)) * (1 - r_c)**2
        ln_gamma2 = (W / (R * T_ref)) * (r_c)**2
        ln_a1 = torch.log(r_c) + ln_gamma1
        ln_a2 = torch.log(1 - r_c) + ln_gamma2
        denom1 = torch.clamp(1.0 - (R / d1) * ln_a1, min=0.1)
        denom2 = torch.clamp(1.0 - (R / d2) * ln_a2, min=0.1)
        Tm_pred = torch.max(T1_t / denom1, T2_t / denom2).numpy()
        mae = _mean_absolute_error(Tm_actual, Tm_pred)
        return mae


def predict_melting_curve(model, x1_emb, x2_emb, T1, T2, R=8.314):
    model.eval()
    with torch.no_grad():
        x1_t = torch.tensor(x1_emb.astype(np.float32)).reshape(1, -1)
        x2_t = torch.tensor(x2_emb.astype(np.float32)).reshape(1, -1)
        d1, d2, W = model(x1_t, x2_t)
        r_range = torch.linspace(0.001, 0.999, 100)
        T_ref = (T1 + T2) / 2.0
        ln_gamma1 = (W / (R * T_ref)) * (1.0 - r_range)**2
        ln_gamma2 = (W / (R * T_ref)) * (r_range)**2
        ln_a1 = torch.log(r_range) + ln_gamma1
        ln_a2 = torch.log(1.0 - r_range) + ln_gamma2
        denom1 = torch.clamp(1.0 - (R / d1) * ln_a1, min=0.1)
        denom2 = torch.clamp(1.0 - (R / d2) * ln_a2, min=0.1)
        Tm_curve = torch.max(T1 / denom1, T2 / denom2)
        return (r_range.numpy(), 
                Tm_curve.numpy(), 
                d1.item(), 
                d2.item(), 
                W.item())
