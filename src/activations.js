export const oblixActivations = {
  apply: function (x, activation) {
    const alpha = 0.01;
    const softplus = (v) => Math.log(1 + Math.exp(v));
    switch (activation) {
      case "tanh":
        return Math.tanh(x);
      case "sigmoid":
        return 1 / (1 + Math.exp(-x));
      case "relu":
        return Math.max(0, x);
      case "leakyrelu":
        return x > 0 ? x : alpha * x;
      case "gelu": {
        const k = 0.7978845608; // sqrt(2/pi)
        const x3 = x * x * x;
        return 0.5 * x * (1 + Math.tanh(k * (x + 0.044715 * x3)));
      }
      case "selu":
        const sa = 1.67326,
          ss = 1.0507;
        return x > 0 ? ss * x : ss * sa * (Math.exp(x) - 1);
      case "swish":
        return x / (1 + Math.exp(-x));
      case "mish":
        return x * Math.tanh(softplus(x));
      case "softmax":
      case "none":
      default:
        return x;
    }
  },
  derivative: function (x, activation) {
    const alpha = 0.01;
    let val;
    const sigmoid = (v) => 1 / (1 + Math.exp(-v));
    const softplus = (v) => Math.log(1 + Math.exp(v));
    const dtanh_dx = (v) => 1 - Math.tanh(v) ** 2;
    switch (activation) {
      case "tanh":
        val = Math.tanh(x);
        return 1 - val * val;
      case "sigmoid":
        val = sigmoid(x);
        return val * (1 - val);
      case "relu":
        return x > 0 ? 1 : 0;
      case "leakyrelu":
        return x > 0 ? 1 : alpha;
      case "gelu": {
        const k = 0.7978845608; // sqrt(2/pi)
        const x2 = x * x;
        const inner = k * (x + 0.044715 * x * x2);
        const tanh_inner = Math.tanh(inner);
        const sech_sq = 1 - tanh_inner * tanh_inner;
        const d_inner_dx = k * (1 + 0.134145 * x2);
        return 0.5 * (1 + tanh_inner) + 0.5 * x * sech_sq * d_inner_dx;
      }
      case "selu":
        const sa = 1.67326,
          ss = 1.0507;
        return x > 0 ? ss : ss * sa * Math.exp(x);
      case "swish":
        const sig_x = sigmoid(x);
        return sig_x + x * sig_x * (1 - sig_x);
      case "mish":
        const sp_x = softplus(x);
        const tanh_sp_x = Math.tanh(sp_x);
        const dsp_dx = sigmoid(x);
        const dtanh_dsp = dtanh_dx(sp_x);
        return tanh_sp_x + x * dtanh_dsp * dsp_dx;
      case "softmax":
      case "none":
      default:
        return 1;
    }
  },
};
