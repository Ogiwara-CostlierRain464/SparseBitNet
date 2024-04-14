from typing import Tuple

from mpmath import ones

from tensorrt_llm import Module, Parameter, Tensor, functional
from tensorrt_llm.layers import Linear, RmsNorm


class BitLinear(Module):
    def __init__(self, in_features, out_features, bias=False, rms_norm_eps=1e-6, bits=8, flg_before_linear=True):
        #super(BitLinear, self).__init__(in_features, out_features, bias)
        super().__init__()
        self.layer_norm = RmsNorm(hidden_size=in_features, eps=rms_norm_eps)
        self.bits = bits
        self.Qb = 2 ** (self.bits - 1)
        self.flg_before_linear = flg_before_linear
        self.epsilon = 1e-6  # overflow防止のための小さな値

    def absmax_quantize(self, x: Tensor, Qb: int, epsilon: float) -> Tuple[Tensor, float]:
        if self.flg_before_linear:
            # パターン①：　通常は[-Qb, Qb]にスケール: 式(4), (5)を適用
            gamma = x.abs().max().clamp(min=epsilon)
            x_scaled: Tensor = x * Qb / gamma
            x_q = functional.round(x_scaled).clamp(-Qb, Qb - 1)
        else:
            # パターン②：　Reluなどの非線形関数前の場合は[0, Qb]にスケール：　式(6)を適用
            # 論文中には記載はないですが、スケールが異なるためスケーリングの基準として使っているgammaもetaを反映した値にすべきだと考えます。
            eta = functional.minimum(x)
            gamma = functional.abs(x - eta).max().clamp(min=epsilon)
            x_scaled = (x - eta) * Qb / gamma
            x_q = functional.round(x_scaled).clamp(0, Qb - 1)
        # STE
        #x_q = (x_q - x_scaled).detach() + x_scaled
        return x_q, gamma

    # 独自のsign関数の定義
    # torch.signは0を0として扱ってしまう。custom_signはW>0を+1に、W≦0を-1とする。
    def custom_sign(self, x: Tensor) -> Tensor:
        gt: Tensor = x > 0
        mu = gt * 2
        m = mu - 1
        return m

    def quantize_weights(self, weight: Tensor, epsilon: float) -> Tuple[Tensor, float]:
        # 式(3): alphaの計算
        # TODO: check
        alpha = weight.mean(dim=0, keepdim=False).mean(dim=1, keepdim=False)

        # 式(1),(2): 重みの中心化とバイナリ化
        weight_centered = weight - alpha
        weight_binarized = self.custom_sign(weight_centered)

        # 式(12): betaの計算
        beta = weight.abs().mean()

        # STE (weight_binarizedとスケールを合わせるためweight_centeredをweight_scaledにスケールしています。)
        # weight_scaled = weight_centered / (weight_centered.abs().max().clamp(min=epsilon))
        # weight_binarized = (weight_binarized - weight_scaled).detach() + weight_scaled

        return weight_binarized, beta

    def forward(self, x: Tensor):
        # 1. LayerNorm (input: x, output: x_norm)
        x_norm = self.layernorm(x)

        # 2. Absmax Quatization (input: x_norm, output: x_q, gamma)
        x_q, gamma = self.absmax_quantize(x_norm, self.Qb, self.epsilon)

        # 3. 1-bit Weights化 (input: -, output: w_q, beta)
        w_q, beta = self.quantize_weights(self.weight, self.epsilon)

        # 4. テンソル積(⊗) (input: x_q,w_q, output: x_matmul)
        x_matmul = torch.nn.functional.linear(x_q, w_q, self.bias)

        # 5. Dequantization (input: x_matmul,beta,gamma, output: output)
        output = x_matmul * (beta * gamma / self.Qb)

        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, flg_before_linear={self.flg_before_linear}'