import math

import torch
import torch.nn as nn


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), self.activation,
        )
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), self.activation,
        )
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        )
        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1), self.activation,
        )

    def forward(self, x):
        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        hx = self.activation(self.en_layer1_4(hx) + hx)
        residual_1 = hx
        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        hx = self.activation(self.en_layer2_4(hx) + hx)
        residual_2 = hx
        hx = self.en_layer3_1(hx)

        return hx, residual_1, residual_2


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.de_layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(320, 192, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(192 + 128, 192, kernel_size=1, padding=0), self.activation,
        )

        head_num = 3
        dim = 192

        self.de_block_1 = HVWA(dim, head_num)
        self.de_block_2 = ECA(dim, head_num)
        self.de_block_3 = CVit(dim, head_num)

        self.de_layer2_1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1), self.activation
        )

    def forward(self, x, residual_1, residual_2):
        hx = self.de_layer3_1(x)
        hx = self.de_layer2_2(torch.cat((hx, residual_2), dim=1))
        fx = hx
        hx = self.de_block_1(hx)
        hx = self.de_block_2(hx, fx)
        hx = self.de_block_3(hx, fx)
        hx = self.de_layer2_1(hx)
        hx = self.activation(self.de_layer1_3(torch.cat((hx, residual_1), dim=1)) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.de_layer1_1(hx)

        return hx


class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class CPE(nn.Module):
    def __init__(self, hidden_size):
        super(CPE, self).__init__()
        self.CPE = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size
        )

    def forward(self, x):
        x = self.CPE(x) + x
        return x


class HWA(nn.Module):
    def __init__(self, dim, head_num):
        super(HWA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.CPE = CPE(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C // 2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C // 2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]
        if H == W:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = (
                attention_output_h.view(B, H, W, C // 2)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            attention_output_v = (
                attention_output_v.view(B, W, H, C // 2)
                .permute(0, 3, 2, 1)
                .contiguous()
            )
            attn_out = self.fuse_out(
                torch.cat((attention_output_h, attention_output_v), dim=1)
            )
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
            attention_output_h = (
                attention_output_h.view(B, H, W, C // 2)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            attention_output_v = (
                attention_output_v.view(B, W, H, C // 2)
                .permute(0, 3, 2, 1)
                .contiguous()
            )
            attn_out = self.fuse_out(
                torch.cat((attention_output_h, attention_output_v), dim=1)
            )
        x = attn_out + h
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)
        x = self.CPE(x)
        return x


class HVWA(nn.Module):
    def __init__(self, dim, head_num):
        super(HVWA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(
            self.hidden_size, self.hidden_size, kernel_size=1, padding=0
        )
        self.conv_h = nn.Conv2d(
            self.hidden_size, 3 * self.hidden_size, kernel_size=1, padding=0
        )  # qkv_h
        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(self.hidden_size)
        self.fuse_out = nn.Conv2d(
            self.hidden_size * 2, self.hidden_size, kernel_size=1, padding=0
        )
        self.attn = Attention(head_num=self.head_num)
        self.CPE = CPE(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)
        x_input_h = torch.chunk(self.conv_input(x), 4, dim=2)
        H_ = H // 4
        C_ = C * 3
        feature_h_1 = torch.chunk(
            (self.conv_h(x_input_h[0]).permute(0, 2, 3, 1).contiguous()).view(
                B * H_, W, C_
            ),
            3,
            dim=2,
        )
        feature_h_2 = torch.chunk(
            (self.conv_h(x_input_h[1]).permute(0, 2, 3, 1).contiguous()).view(
                B * H_, W, C_
            ),
            3,
            dim=2,
        )
        feature_h_3 = torch.chunk(
            (self.conv_h(x_input_h[2]).permute(0, 2, 3, 1).contiguous()).view(
                B * H_, W, C_
            ),
            3,
            dim=2,
        )
        feature_h_4 = torch.chunk(
            (self.conv_h(x_input_h[3]).permute(0, 2, 3, 1).contiguous()).view(
                B * H_, W, C_
            ),
            3,
            dim=2,
        )
        query_h_1, key_h_1, value_h_1 = feature_h_1[0], feature_h_1[1], feature_h_1[2]
        query_h_2, key_h_2, value_h_2 = feature_h_2[0], feature_h_2[1], feature_h_2[2]
        query_h_3, key_h_3, value_h_3 = feature_h_3[0], feature_h_3[1], feature_h_3[2]
        query_h_4, key_h_4, value_h_4 = feature_h_4[0], feature_h_4[1], feature_h_4[2]

        x_input_v = torch.chunk(self.conv_input(x), 4, dim=3)
        W_ = W // 4
        feature_v_1 = torch.chunk(
            (self.conv_h(x_input_v[0]).permute(0, 3, 2, 1).contiguous()).view(
                B * W_, H, C_
            ),
            3,
            dim=2,
        )
        feature_v_2 = torch.chunk(
            (self.conv_h(x_input_v[1]).permute(0, 3, 2, 1).contiguous()).view(
                B * W_, H, C_
            ),
            3,
            dim=2,
        )
        feature_v_3 = torch.chunk(
            (self.conv_h(x_input_v[2]).permute(0, 3, 2, 1).contiguous()).view(
                B * W_, H, C_
            ),
            3,
            dim=2,
        )
        feature_v_4 = torch.chunk(
            (self.conv_h(x_input_v[3]).permute(0, 3, 2, 1).contiguous()).view(
                B * W_, H, C_
            ),
            3,
            dim=2,
        )
        query_v_1, key_v_1, value_v_1 = feature_v_1[0], feature_v_1[1], feature_v_1[2]
        query_v_2, key_v_2, value_v_2 = feature_v_2[0], feature_v_2[1], feature_v_2[2]
        query_v_3, key_v_3, value_v_3 = feature_v_3[0], feature_v_3[1], feature_v_3[2]
        query_v_4, key_v_4, value_v_4 = feature_v_4[0], feature_v_4[1], feature_v_4[2]
        attn_output_h1 = (
            self.attn(query_h_1, key_h_1, value_h_1)
            .view(B, H_, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_h2 = (
            self.attn(query_h_2, key_h_2, value_h_2)
            .view(B, H_, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_h3 = (
            self.attn(query_h_3, key_h_3, value_h_3)
            .view(B, H_, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_h4 = (
            self.attn(query_h_4, key_h_4, value_h_4)
            .view(B, H_, W, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_h = torch.cat(
            (attn_output_h1, attn_output_h2, attn_output_h3, attn_output_h4), dim=2
        )
        attn_output_1 = (
            self.attn(query_v_1, key_v_1, value_v_1)
            .view(B, H, W_, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_2 = (
            self.attn(query_v_2, key_v_2, value_v_2)
            .view(B, H, W_, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_3 = (
            self.attn(query_v_3, key_v_3, value_v_3)
            .view(B, H, W_, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_4 = (
            self.attn(query_v_4, key_v_4, value_v_4)
            .view(B, H, W_, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        attn_output_v = torch.cat(
            (attn_output_1, attn_output_2, attn_output_3, attn_output_4), dim=3
        )
        attn_out = self.fuse_out(torch.cat((attn_output_h, attn_output_v), dim=1))
        x = attn_out + h
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)
        x = self.CPE(x)
        return x


class ECA(nn.Module):
    def __init__(self, dim, head_num):
        super(ECA, self).__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.hidden_size = dim
        self.head_num = head_num
        self.attn_norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.de_layer = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            self.activation,
        )
        self.q_local_block = nn.Linear(self.hidden_size, self.hidden_size)
        self.kv_local_block = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.CPE = CPE(dim)

    def forward(self, x, fx):
        hx = fx
        x = self.conv(x)
        fx = self.conv(fx)
        h = fx
        B, C, H, W = fx.size()
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()
        x = self.attn_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)
        x_input = self.conv_input(x).permute(0, 2, 3, 1).contiguous()
        x_input = x_input.view(B, H * W, C)
        q = self.q_local_block(x_input)
        fx = fx.view(B, C, H * W).permute(0, 2, 1).contiguous()
        fx = self.attn_norm(fx).permute(0, 2, 1).contiguous()
        fx = fx.view(B, C, H, W)
        fx_input = self.conv_input(fx).permute(0, 2, 3, 1).contiguous()
        fx_input = fx_input.view(B, H * W, C)
        kv = torch.chunk(self.kv_local_block(fx_input), 2, dim=-1)
        k, v = kv[0], kv[1]
        attention_output = self.attn(q, k, v)
        attention_output = (
            attention_output.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        )
        attn_out = self.fuse_out(attention_output)
        fx = attn_out + h
        fx = fx.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = fx
        fx = self.ffn_norm(fx)
        fx = self.ffn(fx)
        fx = fx + h
        fx = fx.permute(0, 2, 1).contiguous()
        fx = fx.view(B, C, H, W)
        fx = self.CPE(fx)
        fx = self.de_layer(fx)
        return fx + hx


class CVit(nn.Module):
    def __init__(self, dim, head_num):
        super(CVit, self).__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.hidden_size = dim
        self.head_num = head_num
        self.attn_norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.de_layer = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1),
            self.activation,
        )
        self.qkv_local_block = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.attn = Attention(head_num=self.head_num)
        self.CPE = CPE(dim)

    def forward(self, x, fx):
        f = x
        x = self.conv(x)
        hx = x
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1).contiguous()  # x: B HW C
        x = self.attn_norm(x).permute(0, 2, 1).contiguous()  # B C HW
        x = x.view(B, C, H, W)
        x_input = self.conv_input(x).permute(0, 2, 3, 1).contiguous()
        x_input = x_input.view(B, H * W, C)
        qkv = torch.chunk(self.qkv_local_block(x_input), 3, dim=-1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention_output = self.attn(q, k, v)
        attention_output = (
            attention_output.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        )
        attn_out = self.fuse_out(attention_output)
        fx = attn_out + hx
        fx = fx.view(B, C, H * W).permute(0, 2, 1).contiguous()
        h = fx
        fx = self.ffn_norm(fx)
        fx = self.ffn(fx)
        fx = fx + h
        fx = fx.permute(0, 2, 1).contiguous()
        fx = fx.view(B, C, H, W)
        fx = self.CPE(fx)
        fx = self.de_layer(fx)
        return fx + f


class ECT(nn.Module):
    def __init__(self):
        super(ECT, self).__init__()
        self.encoder = DownSample()
        head_num = 5
        dim = 320
        self.Trans_block_1_1 = HVWA(dim, head_num)
        self.Trans_block_1_2 = ECA(dim, head_num)
        self.Trans_block_2_1 = HVWA(dim, head_num)
        self.Trans_block_2_2 = ECA(dim, head_num)
        self.Trans_block_2_3 = CVit(dim, head_num)
        self.Trans_block_3_1 = HVWA(dim, head_num)
        self.Trans_block_3_2 = ECA(dim, head_num)
        self.Trans_block_4_1 = HVWA(dim, head_num)
        self.Trans_block_4_2 = ECA(dim, head_num)
        self.decoder = UpSample()

    def forward(self, x):
        hx, residual_1, residual_2 = self.encoder(x)
        h1 = hx
        hx = self.Trans_block_1_1(hx)
        hx = self.Trans_block_1_2(hx, h1)
        hx = self.Trans_block_2_1(hx)
        hx = self.Trans_block_2_2(hx, h1)
        hx = self.Trans_block_2_3(hx, h1)
        hx = self.Trans_block_3_1(hx)
        hx = self.Trans_block_3_2(hx, h1)
        hx = self.Trans_block_4_1(hx)
        hx = self.Trans_block_4_2(hx, h1)
        hx = self.decoder(hx, residual_1, residual_2)
        return hx + x
